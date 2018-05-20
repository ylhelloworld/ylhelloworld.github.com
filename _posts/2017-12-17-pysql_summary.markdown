---
layout:     post
title:      "Pymssql Summary"
subtitle:   "pymssql basic feature & _mssql"
date:       2017-12-17 12:00:00
author:     "YangLong"
header-img: "img/post-bg-nextgen-web-pwa.jpg"
header-mask: 0.3
catalog:    true
tags:
    - Python
    - SqlServer 
---

### Intruction  
*The pymssql package consists of two modules:  
•pymssql – use it if you care about DB-API compliance, or if you are accustomed to DB-API syntax  
•_mssql – use it if you care about performance and ease of use (_mssql module is easier to use than pymssql).*  
![架构图](http://pymssql.org/en/stable/_images/pymssql-stack.png)

### Examples
#### pymssql
##### basic features
```python 
from os import getenv
import pymssql

server = getenv("PYMSSQL_TEST_SERVER")
user = getenv("PYMSSQL_TEST_USERNAME")
password = getenv("PYMSSQL_TEST_PASSWORD")

conn = pymssql.connect(server, user, password, "tempdb")
cursor = conn.cursor()
cursor.execute("""
IF OBJECT_ID('persons', 'U') IS NOT NULL
    DROP TABLE persons
CREATE TABLE persons (
    id INT NOT NULL,
    name VARCHAR(100),
    salesrep VARCHAR(100),
    PRIMARY KEY(id)
)
""")
cursor.executemany(
    "INSERT INTO persons VALUES (%d, %s, %s)",
    [(1, 'John Smith', 'John Doe'),
     (2, 'Jane Doe', 'Joe Dog'),
     (3, 'Mike T.', 'Sarah H.')])
# you must call commit() to persist your data if you don't set autocommit to True
conn.commit()

cursor.execute('SELECT * FROM persons WHERE salesrep=%s', 'John Doe')
row = cursor.fetchone()
while row:
    print("ID=%d, Name=%s" % (row[0], row[1]))
    row = cursor.fetchone()

conn.close()

```

#### _mssql
##### execute_non_query
```python 
import _mssql
conn = _mssql.connect(server='SQL01', user='user', password='password', \
    database='mydatabase')
conn.execute_non_query('CREATE TABLE persons(id INT, name VARCHAR(100))')
conn.execute_non_query("INSERT INTO persons VALUES(1, 'John Doe')")
conn.execute_non_query("INSERT INTO persons VALUES(2, 'Jane Doe')")
conn.close()
```
##### execute_query
```python
# how to fetch rows from a table
conn.execute_query('SELECT * FROM persons WHERE salesrep=%s', 'John Doe')
for row in conn:
    print "ID=%d, Name=%s" % (row['id'], row['name'])
```
##### execute_query_scalar & execute_query_row
```python
# examples of other query functions
numemployees = conn.execute_scalar("SELECT COUNT(*) FROM employees")
numemployees = conn.execute_scalar("SELECT COUNT(*) FROM employees WHERE name LIKE 'J%'")    # note that '%' is not a special character here
employeedata = conn.execute_row("SELECT * FROM employees WHERE id=%d", 13)

```
##### procedure
```python
# how to fetch rows from a stored procedure
conn.execute_query('sp_spaceused')   # sp_spaceused without arguments returns 2 result sets
res1 = [ row for row in conn ]       # 1st result
res2 = [ row for row in conn ]       # 2nd result
```
```python
# how to get an output parameter from a stored procedure
sqlcmd = """
DECLARE @res INT
EXEC usp_mystoredproc @res OUT
SELECT @res
"""
res = conn.execute_scalar(sqlcmd)
```
```python
# how to get more output parameters from a stored procedure
sqlcmd = """
DECLARE @res1 INT, @res2 TEXT, @res3 DATETIME
EXEC usp_getEmpData %d, %s, @res1 OUT, @res2 OUT, @res3 OUT
SELECT @res1, @res2, @res3
"""
res = conn.execute_row(sqlcmd, (13, 'John Doe'))
```
```python
# examples of queries with parameters
conn.execute_query('SELECT * FROM empl WHERE id=%d', 13)
conn.execute_query('SELECT * FROM empl WHERE name=%s', 'John Doe')
conn.execute_query('SELECT * FROM empl WHERE id IN (%s)', ((5, 6),))
conn.execute_query('SELECT * FROM empl WHERE name LIKE %s', 'J%')
conn.execute_query('SELECT * FROM empl WHERE name=%(name)s AND city=%(city)s', \
    { 'name': 'John Doe', 'city': 'Nowhere' } )
conn.execute_query('SELECT * FROM cust WHERE salesrep=%s AND id IN (%s)', \
    ('John Doe', (1, 2, 3)))
conn.execute_query('SELECT * FROM empl WHERE id IN (%s)', (tuple(xrange(4)),))
conn.execute_query('SELECT * FROM empl WHERE id IN (%s)', \
    (tuple([3, 5, 7, 11]),))
```
##### exception handler 
*You can provide your own message handler callback function that will be invoked by the stack with informative messages sent by the server. Set it on a per _mssql connection basis by using the _mssql.MSSQLConnection.set_msghandler() method:*
```python
import _mssql

def my_msg_handler(msgstate, severity, srvname, procname, line, msgtext):
    """
    Our custom handler -- It simpy prints a string to stdout assembled from
    the pieces of information sent by the server.
    """
    print("my_msg_handler: msgstate = %d, severity = %d, procname = '%s', "
          "line = %d, msgtext = '%s'" % (msgstate, severity, procname,
                                         line, msgtext))

conn = _mssql.connect(server='SQL01', user='user', password='password')
try:
    conn.set_msghandler(my_msg_handler)  # Install our custom handler
    cnx.execute_non_query("USE mydatabase")  # It gets called at this point
finally:
    conn.close()
```