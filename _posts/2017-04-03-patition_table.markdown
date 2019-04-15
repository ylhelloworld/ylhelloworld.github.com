---
layout:     post
title:      "SqlServer分区表"
subtitle:   "SQL Server Partition Table"
date:       2017-04-03 12:00:00
author:     "YangLong"
header-img: "img/spiderman/post04.jpg"
header-mask: 0.3
catalog:    true 
multilingual: false  
tags:
    - SqlServer 
    - Database
---


## SqlServer 分区表  

#### Partition Function 
-  Create Partition Function
```sql
create partition function 分区函数名(<分区列类型>) as range [left/right] 
for values (每个分区的边界值,....) 
	
--demo
CREATE PARTITION FUNCTION [demoPartitionFun](int) AS RANGE LEFT FOR VALUES (N'1000000', N'2000000', N'3000000', N'4000000', N'5000000', N'6000000', N'7000000', N'8000000', N'9000000', N'10000000')
```

-  Delete Partition Function  
```sql
drop partition function <分区函数名>
-- demo
--删除分区函数 demoPartitionFun
drop partition function demoPartitionFun

```

#### Partition Scheme  
- Create Patition Scheme  
```sql
create partition scheme <分区方案名称> as partition <分区函数名称> [all]to (文件组名称,....) 
--demo
--创建分区方案,所有分区在一个组里面
CREATE PARTITION SCHEME [demoPartitionSchema] AS PARTITION [demoPartitionFun] TO ([ByIdGroup1], [ByIdGroup1], [ByIdGroup1], [ByIdGroup1], [ByIdGroup1], [ByIdGroup1], [ByIdGroup1], [ByIdGroup1], [ByIdGroup1], [ByIdGroup1], [ByIdGroup1])

```

- Delete Partition Scheme  
```sql
drop partition scheme<分区方案名称>

--demo
--删除分区方案 demoPartitionSchema
drop partition scheme demoPartitionSchema1
``` 

#### Create Partiiton Table
```sql
create table <表名> (
  <列定义>
)on<分区方案名>(分区列名)

--demo
--创建分区表
create table BigOrder (
   OrderId              int                  identity,
   orderNum             varchar(30)          not null,
   OrderStatus          int                  not null default 0,
   OrderPayStatus       int                  not null default 0,
   UserId               varchar(40)          not null,
   CreateDate           datetime             null default getdate(),
   Mark                 nvarchar(300)        null
)on bgPartitionSchema(OrderId)

```

#### Create Partition Index
```sql 
create <索引分类> index <索引名称> 
on <表名>(列名)
on <分区方案名>(分区依据列名)

--创建分区索引
CREATE CLUSTERED INDEX [ClusteredIndex_on_bgPartitionSchema_635342971076448165] ON [dbo].[BigOrder] 
(
    [OrderId]
)WITH (SORT_IN_TEMPDB = OFF, IGNORE_DUP_KEY = OFF, DROP_EXISTING = OFF, ONLINE = OFF) ON [bgPartitionSchema]([OrderId])

```


#### Partition Statistics 
- query the data where patition  location
```sql
--查询分区依据列为10000014的数据在哪个分区上
select $partition.bgPartitionFun(2000000)  --返回值是2，表示此值存在第2个分区 

```
- the number of each partition
```sql
--查看分区表中，每个非空分区存在的行数
select $partition.bgPartitionFun(orderid) as partitionNum,count(*) as recordCount
from bigorder
group by  $partition.bgPartitionFun(orderid)

```
- the record of the special partition 
```sql 
---查看指定分区中的数据记录
select * from bigorder where $partition.bgPartitionFun(orderid)=2
```

#### Split&Merge
- Split Partition
```sql 
--分区拆分
alter partition function bgPartitionFun()
split range(N'1500000')  --将第二个分区拆为2个分区

```

- Merge Partition
```sql 
--合并分区
alter partition function bgPartitionFun()
merge range(N'1500000')  --将第二第三分区合并

```
- Move Partition 
```sql 

--分区到表
--将bigorder分区表中的第一分区数据复制到普通表中
alter table bigorder switch partition 1 to <普通表名>
--表到分区
--将普通表中的数据复制到bigorder分区表中的第一分区
alter table <普通表名> switch to bigorder partition 1 
```




---
