# OdinVectorDB
什么是OdinVectorDB？一个功能强大的向量数据库原型：
* **低内存开销**：全流程系统内存峰维持80G以内；
* **支持带标签约束的ANN**：支持多标签向量近似近邻混合检索；
* **高可维护性**：支持索引高效动态维护（插入、删除、更新）；
* **兼容原生ANN**：支持基于SSD的近似近邻向量检索；
* **高可用性**：支持基于WAL的崩溃恢复功能。

## 编译与安装
* 安装cmake、boost等基本依赖库
* 安装third-party中的liburing
* 安装CRoaring库
* 配置CMakeLists.txt指向系统中的CRoaring库
* 整体编译安装

## 主要可用工具
* build_disk_index：用于构建基本图索引

* search_disk_index：用于原生ANN查询

* build_ivf_index：用于构建IVF索引，用于混合检索

* hybrid_tag_search：用于混合检索

* test_insert：用于测试插入删除更新向量等功能

* test_dynamic_ivf：用于测试IVF索引的动态维护功能

* test_insert_hybrid：待实现，用于测试混合检索中插入删除更新向量等功能

