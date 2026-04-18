import numpy as np
import struct
import sys
import os

global cur_data_type
cur_data_type = "float32"
cur_data_type_size = 4

# bin format:
# | 4 bytes for num_vecs | 4 bytes for vector dimension (e.g., 100 for SPACEV) | flattened vectors |
def bin_write(vectors, filename):
    with open(filename, 'wb') as f:
        num_vecs, vector_dim = vectors.shape
        f.write(struct.pack('<i', num_vecs))
        f.write(struct.pack('<i', vector_dim))
        f.write(vectors.tobytes())

def bin_read_metadata(filename):
    with open(filename, 'rb') as f:
        num_vecs = struct.unpack('<i', f.read(4))[0]
        vector_dim = struct.unpack('<i', f.read(4))[0]
        return {
            "num_vecs": num_vecs,
            "vector_dim": vector_dim
        }

def bin_read_vectors(filename, indexes):
    # read vectors with given indexes from binary file, without read the whole data.
    ret = []
    with open(filename, 'rb') as f:
        num_vecs = struct.unpack('<i', f.read(4))[0]
        vector_dim = struct.unpack('<i', f.read(4))[0]

        for index in indexes:
            f.seek(index * vector_dim * cur_data_type_size + 8)
            data = f.read(vector_dim * cur_data_type_size)
            vectors = np.frombuffer(data, dtype=cur_data_type).reshape((1, vector_dim))
            ret.append(vectors)
    return ret

"""
C++ code for reading metadata page:
in.read((char *) &nr, sizeof(_u32));
in.read((char *) &nc, sizeof(_u32));

in.read((char *) &npoints, sizeof(_u64));
in.read((char *) &data_dim, sizeof(_u64));

in.read((char *) &entry_point, sizeof(_u64));
in.read((char *) &max_node_len, sizeof(_u64));
in.read((char *) &nnodes_per_sector, sizeof(_u64));
"""

# metadata page: 
def index_read_metadata(filename):
    with open(filename, 'rb') as f:
        nr = struct.unpack('<I', f.read(4))[0]
        nc = struct.unpack('<I', f.read(4))[0]
        npoints = struct.unpack('<Q', f.read(8))[0]
        data_dim = struct.unpack('<Q', f.read(8))[0]
        entry_point = struct.unpack('<Q', f.read(8))[0]
        max_node_len = struct.unpack('<Q', f.read(8))[0]
        nnodes_per_sector = struct.unpack('<Q', f.read(8))[0]
    print(f"npoints: {npoints}, data_dim: {data_dim}, entry_point: {entry_point}, max_node_len: {max_node_len}, nnodes_per_sector: {nnodes_per_sector}")
    return {
        "nr": nr,
        "nc": nc,
        "npoints": npoints,
        "data_dim": data_dim,
        "entry_point": entry_point,
        "max_node_len": max_node_len,
        "nnodes_per_sector": nnodes_per_sector
    }

def index_read_vectors(filename, indexes, metadata):
    # in each page (except the first), there are nnodes_per_sector records, each record is max_node_len.
    # The vector resides in the record's [0, data_dim * size_per_var]
    ret = []
    print(f"cur_data_type: {cur_data_type}, itemsize: {cur_data_type_size}")
    for index in indexes:
        page_index = 1 + index // metadata["nnodes_per_sector"]
        record_index = index % metadata["nnodes_per_sector"]
        with open(filename, 'rb') as f:
            f.seek(page_index * 4096 + record_index * metadata["max_node_len"])
            vec_data = f.read(metadata["data_dim"] * cur_data_type_size)
            vec = np.frombuffer(vec_data, dtype=cur_data_type).reshape((1, metadata["data_dim"]))
            ret.append(vec)
    return ret


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python validate_index_structure.py <data_type> <index_file> <vector_bin>")
        sys.exit(1)

    data_type = sys.argv[1]
    index_file = sys.argv[2]
    vec_bin_file = sys.argv[3]
    
    if data_type == "float":
        cur_data_type = "float32"
        cur_data_type_size = 4
    elif data_type == "uint8":
        cur_data_type = "uint8"
        cur_data_type_size = 1
    elif data_type == "int8":
        cur_data_type = "int8"
        cur_data_type_size = 1
    else:
        print(f"Error: Invalid data type: {data_type}")
        sys.exit(1)

    # read metadata from bin file
    vec_metadata = bin_read_metadata(vec_bin_file)

    # Read metadata page
    metadata = index_read_metadata(index_file)
    
    # validate metadata
    if metadata["npoints"] != vec_metadata["num_vecs"]:
        print(f"Error: Number of vectors mismatch. Expected {metadata['npoints']}, got {vec_metadata['num_vecs']}")
        sys.exit(1)
    if metadata["data_dim"] != vec_metadata["vector_dim"]:
        print(f"Error: Vector dimension mismatch. Expected {metadata['data_dim']}, got {vec_metadata['vector_dim']}")
        sys.exit(1)
    print("Metadata is valid.")

    indexes = np.random.randint(0, metadata["npoints"], 100)
    print(f"Random checking indexes: {indexes}")
    vectors = index_read_vectors(index_file, indexes, metadata)
    vectors2 = bin_read_vectors(vec_bin_file, indexes)
    for i in range(len(indexes)):
        if not np.allclose(vectors[i], vectors2[i]):
            print(f"Error: Vector mismatch at index {indexes[i]}. Expected {vectors2[i]}, got {vectors[i]}")
            sys.exit(1)
    print("Index file is valid.")

    print("Validating index file size...")
    # validate index file size.
    index_size = os.path.getsize(index_file)
    n_sectors = 1 + metadata["npoints"] // metadata["nnodes_per_sector"] + (metadata["npoints"] % metadata["nnodes_per_sector"] > 0)
    if index_size != 4096 * n_sectors:
        print(f"Error: Index file size mismatch. Expected {4096 * n_sectors}, got {index_size}")
        # read the N + 1 vector
        vec = index_read_vectors(index_file, [metadata["npoints"]], metadata)
        print(f"{metadata['npoints'] + 1}th vector: {vec[0]}")
        sys.exit(1)
    
    print(f"Index size is valid, expected {4096 * n_sectors / 1024 / 1024 / 1024} GB, got {index_size / 1024 / 1024 / 1024} GB")
