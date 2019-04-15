---
layout:     post
title:      "C# 语言变更历史"
subtitle:   "C# language edtion history"
date:       2017-03-27 12:00:00
author:     "YangLong"
header-img: "img/spiderman/post02.jpg"
header-mask: 0.3
catalog:    true 
multilingual: false  
tags:
    - C#.Net Framework 
---



## .Net  Framework 的版本信息 

<html>
<table>
<thead>
<tr>
<td>.NET Standard</td>
<td><a href="https://github.com/dotnet/standard/blob/master/docs/versions/netstandard1.0.md" data-linktype="external">1.0</a></td>
<td><a href="https://github.com/dotnet/standard/blob/master/docs/versions/netstandard1.1.md" data-linktype="external">1.1</a></td>
<td><a href="https://github.com/dotnet/standard/blob/master/docs/versions/netstandard1.2.md" data-linktype="external">1.2</a></td>
<td><a href="https://github.com/dotnet/standard/blob/master/docs/versions/netstandard1.3.md" data-linktype="external">1.3</a></td>
<td><a href="https://github.com/dotnet/standard/blob/master/docs/versions/netstandard1.4.md" data-linktype="external">1.4</a></td>
<td><a href="https://github.com/dotnet/standard/blob/master/docs/versions/netstandard1.5.md" data-linktype="external">1.5</a></td>
<td><a href="https://github.com/dotnet/standard/blob/master/docs/versions/netstandard1.6.md" data-linktype="external">1.6</a></td>
<td><a href="https://github.com/dotnet/standard/blob/master/docs/versions/netstandard2.0.md" data-linktype="external">2.0</a></td>
</tr>
</thead>
<tbody>
<tr>
<td>.NET Core</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
<td>2.0</td>
</tr>
<tr>
<td>.NET Framework <sup>1</sup></td>
<td>4.5</td>
<td>4.5</td>
<td class="x-hidden-focus">4.5.1</td>
<td>4.6</td>
<td>4.6.1</td>
<td>4.6.1</td>
<td>4.6.1</td>
<td>4.6.1</td>
</tr>
<tr>
<td>Mono</td>
<td class="x-hidden-focus">4.6</td>
<td>4.6</td>
<td>4.6</td>
<td>4.6</td>
<td>4.6</td>
<td>4.6</td>
<td>4.6</td>
<td>5.4</td>
</tr>
<tr>
<td>Xamarin.iOS</td>
<td>10.0</td>
<td>10.0</td>
<td>10.0</td>
<td>10.0</td>
<td>10.0</td>
<td>10.0</td>
<td>10.0</td>
<td>10.14</td>
</tr>
<tr>
<td>Xamarin.Mac</td>
<td>3.0</td>
<td>3.0</td>
<td>3.0</td>
<td>3.0</td>
<td>3.0</td>
<td>3.0</td>
<td>3.0</td>
<td>3.8</td>
</tr>
<tr>
<td>Xamarin.Android</td>
<td>7.0</td>
<td>7.0</td>
<td>7.0</td>
<td>7.0</td>
<td>7.0</td>
<td>7.0</td>
<td>7.0</td>
<td>8.0</td>
</tr>
<tr>
<td>Universal Windows Platform</td>
<td>10.0</td>
<td>10.0</td>
<td>10.0</td>
<td>10.0</td>
<td>10.0</td>
<td>10.0.16299</td>
<td>10.0.16299</td>
<td>10.0.16299</td>
</tr>
<tr>
<td>Windows</td>
<td>8.0</td>
<td>8.0</td>
<td>8.1</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>Windows Phone</td>
<td>8.1</td>
<td>8.1</td>
<td>8.1</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>Windows Phone Silverlight</td>
<td>8.0</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
</tbody>
</table>	
</html>


## VERSION 2.0
#### Geenerics *泛型*  

可以在类、方法中对使用的类型进行参数化  
```cs
// Declare the generic class.
public class GenericList<T>
{
    public void Add(T input) { }
}
class TestGenericList
{
    private class ExampleClass { }
    static void Main()
    {
        // Declare a list of type int.
        GenericList<int> list1 = new GenericList<int>();
        list1.Add(1);

        // Declare a list of type string.
        GenericList<string> list2 = new GenericList<string>();
        list2.Add("");

        // Declare a list of type ExampleClass.
        GenericList<ExampleClass> list3 = new GenericList<ExampleClass>();
        list3.Add(new ExampleClass());
    }
}
```
##### 泛型约束  
可以给泛型的类型参数上加约束，可以要求这些类型参数满足一定的条件  
约束 | 说明 
---|---
where T: struct	|类型参数需是值类型
where T : class	|类型参数需是引用类型
where T : new()	|类型参数要有一个public的无参构造函数
where T : <base class name>	|类型参数要派生自某个基类
where T : <interface name>	|类型参数要实现了某个接口
where T : U	|这里T和U都是类型参数，T必须是或者派生自U


#### Partial types  *部分类*  
```cs
public partial class Employee
{
    public void DoWork()
    {
    }
}

public partial class Employee
{
    public void GoToLunch()
    {
    }
}
```

#### Anonymous methods *匿名方法*   
```cs
delegate void Del(int x);
Del d = delegate(int k) { /* ... */ }; 
System.Threading.Thread t1 = new System.Threading.Thread (delegate() { System.Console.Write("Hello, "); } );
```

#### Nullable types *可控类型* 
System.Nullable<T>简写为T ?   
可空类型System.Nullable<T>，可空类型仅针对于值类型，不能针对引用类型去创建

#### Iterators *迭代器*  
```cs
static void Main()  
{  
    DaysOfTheWeek days = new DaysOfTheWeek();  

    foreach (string day in days)  
    {  
        Console.Write(day + " ");  
    }  
    // Output: Sun Mon Tue Wed Thu Fri Sat  
    Console.ReadKey();  
}  

public class DaysOfTheWeek : IEnumerable  
{  
    private string[] days = { "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat" };  

    public IEnumerator GetEnumerator()  
    {  
        for (int index = 0; index < days.Length; index++)  
        {  
            // Yield each day of the week.  
            yield return days[index];  
        }  
    }  
}  
```




#### Covariance and Contravariance *委托的协变和逆变*

```cs
// Assignment compatibility.   
string str = "test";  
// An object of a more derived type is assigned to an object of a less derived type.   
object obj = str;  

// Covariance.   
IEnumerable<string> strings = new List<string>();  
// An object that is instantiated with a more derived type argument   
// is assigned to an object instantiated with a less derived type argument.   
// Assignment compatibility is preserved.   
IEnumerable<object> objects = strings;  

// Contravariance.             
// Assume that the following method is in the class:   
// static void SetObject(object o) { }   
Action<object> actObject = SetObject;  
// An object that is instantiated with a less derived type argument   
// is assigned to an object instantiated with a more derived type argument.   
// Assignment compatibility is reversed.   
Action<string> actString = actObject;  
```  


#### Default *默认类型* 
以使用在类型参数上：default(T);
对于值类型，返回0，引用类型，返回null，对于结构类型，会返回一个成员值全部为0的结构实例  


#### Global *上层空间*   
解决类名重复问题，从最外面逐渐向内部寻找System类  

```cs
global::System.Console.WriteLine(number);
```

#### Fiexd *固定类型*

```cs
public fixed char pathName[128]; 
```

## VERSION 3.0 
#### Lamada表达式（=>)
##### 委托 Delegate
Delegate至少0个参数，至多32个参数，可以无返回值，也可以指定返回值类型
如：
```cs  
public delegate int MethodDelegate(int x, int y);
 private static MethodDelegate method;
 static void Main(string[] args)
 {
            method = new MethodDelegate(Add);
            Console.WriteLine(method(10,20));
            Console.ReadKey();
 }
 private static int Add(int x, int y)
 {
            return x + y;
 }
```
##### Action 无返回值的委托 
一个无返回值、泛型的委托，有16重载，分别是0个参数~15个参数

delegate void Action();
delegate void Action<in T>(T obj);
delegate void Action<in T,in T1>(T obj,T1 obj1);
delegate void Action<in T,in T1,in T2>(T obj,T1 obj1,T2 obj2);
如:
```cs
this.button1.Click += new Action<object, EventArgs>((sender, e) => {
                MessageBox.Show("hellow");
            });
```
如:
```cs
Thread t = new Thread(new ThreadStart(new Action(() => { 
			                //线程代码
			            })));
			            t.Start();
```
##### Func 有返回值委托
带有返回值的委托。也是有15个重载
delegate TR Func(out TR);
delegate TR Func<in T,out TR>(T obj);
delegate TR Func<in T,in T1,out TR>(T obj,T1 obj1);
delegate TR Func<in T,in T1,in T2,out TR>(T obj,T1 obj1,T2 obj2);
```cs
MessageBox.Show(new Func<string>(() => {
                System.Net.WebClient wc = new System.Net.WebClient();
                return wc.DownloadString("http://www.baidu.com");
            })());
```
##### Expression 表达式树
Expression<Func<type,returnType>> = (param) => lamdaexpresion;
```cs
Expression<Func<int, int, int>> expr = (x, y) => x+y;
Expression<Func<int, int, int>> expr = (x, y) => { return x + y; };  //会出现错误，需用API创建复杂的表达式树
```		 
#### 类型推断 （Var）
```cs
var i = 5;
var s = "Hello";
```
				
#### 扩展方法
```cs
public static class JeffClass
{
    public static int StrToInt32(this string s)
    {
        return Int32.Parse(s);
    }
 }
```		
#### 自动属性（get、set）
会自动生成一个后台的私有变量
```cs
public Class Point
{
   public int X { get; set; }
   public int Y { get; set; }
}
``` 

#### 查询表达式
```cs
from g in
from c in customers
           group c by c.Country
select new { Country = g.Key, CustCount = g.Count() }
```
## VERSION 4.0 

#### 斜变和逆变
支持针对泛型接口的协变和逆变
```cs
IList<string> strings = new List<string>();
IList<object> objects = strings;
```
#### 可选参数和命名参数
```cs
private void CreateNewStudent(string name, int studentid = 0, int year = 1)
CreateNewStudent(year:2, name:"Hima", studentid: 4);  
```
## VERSION 5.0
#### 异步编程
如：
```cs
static async void DownloadStringAsync2(Uri uri)
{
    var webClient = new WebClient();
    var result = await webClient.DownloadStringTaskAsync(uri);
    Console.WriteLine(result);
}
```
之前的方法：
```cs
static void DownloadStringAsync(Uri uri)
{
  var webClient = new WebClient();
  webClient.DownloadStringCompleted += (s, e) =>
      {
          Console.WriteLine(e.Result);
      };
  webClient.DownloadStringAsync(uri);
 }
 ```
#### 调用用法信息
```cs
public void DoProcessing()
{
    TraceMessage("Something happened.");
}
```
## VERSION 6.0
#### 自动属性初始化(Auto-property initializers)
```cs
public class Account
{
    public string Name { get; set; } = "summit";
    public int Age { get; set; } = 22;
    public IList<int> AgeList
    {
        get;
        set;
    } = new List<int> { 10,20,30,40,50 };
}
```
#### 字符串嵌入值(String interpolation)
```cs
Console.WriteLine($"年龄:{account.Age}  生日:{account.BirthDay.ToString("yyyy-MM-dd")}");
Console.WriteLine($"年龄:{account.Age}");
```
#### 导入静态类(Using Static)
```cs
using static System.Math;//导入类
Console.WriteLine($"之前的使用方式: {Math.Pow(4, 2)}");
Console.WriteLine($"导入后可直接使用方法: {Pow(4,2)}");
```
#### 空值运算符(Null-conditional operators)
```cs
var age = account.AgeList?[0].ToString();
Console.WriteLine("{0}", (person.list?.Count ?? 0));
```

#### 异常过滤器(Exception filters)
```cs
static void TestExceptionFilter()
{
    try
    {
        Int32.Parse("s");
    }
    catch (Exception e) when (Log(e))
    {
        Console.WriteLine("catch");
        return;
    }
}
```
#### nameof表达式 (nameof expressions)
```cw
private static void Add(Account account)
{
     if (account == null)
         throw new ArgumentNullException("account");
}
```
#### catch和finally语句块里使用await(Await in catch and finally blocks)
```cs
Resource res = null;
try
{
    res = await Resource.OpenAsync(…);       // You could do this.
    …
}
catch(ResourceException e)
{
    await Resource.LogAsync(res, e);         // Now you can do this …
}
finally
{
    if (res != null) await res.CloseAsync(); // … and this.
}
```
#### 属性使用Lambda表达式(Expression bodies on property-like function members)
```cs
public string Name =>string.Format("姓名: {0}", "summit");
public void Print() => Console.WriteLine(Name);
```
#### 方法成员上使用Lambda表达式
```cs
static int LambdaFunc(int x, int y) => x*y;
public void Print() => Console.WriteLine(First + " " + Last);
```
## VERSION 7.0
#### 元祖 Tuple
```cs
( string, string, string, string) getEmpInfo()
{
    //read EmpInfo from database or any other source and just return them
    string strFirstName = "abc";
    string strAddress = "Address";
    string strCity= "City";
    string strState= "State";
    return (strFirstName, strAddress, strCity, strState); // tuple literal
                 return (strFName: strFirstName, strAdd: strAddress, strCity: strC, strState: strSt);
}
var empInfo= getEmpInfo();
```

#### 析构函数 
```cs
( string strFName,  string strAdd,  string strC, string strSt) = getEmpInfo(); 
 Console.WriteLine($"Address: { strAdd }, Country: { strC }");
```
#### OUT输出变量
```cs
 void AssignVal(out string strName)
 {
        strName = "I am from OUT";
 }
  AssignVal(out string szArgu); 
```
#### 不允许为空类型
```cs
int objNullVal;     //non-nullable value type
int? objNotNullVal;    //nullable value type
string! objNotNullRef; //non-nullable reference type
string objNullRef;  //nullable reference type
```
#### 本地方法和函数
```cs
private static void Main(string[] args)
{
    int local_var = 100;
    int LocalFunction(int arg1)
    {
        return local_var * arg1;
    }
 
    Console.WriteLine(LocalFunction(100));
}
```
#### 模式匹配
```cs
class Calculate();
class Add(int a, int b, int c) : Calculate;
class Substract(int a, int b) : Calculate;
class Multiply(int a, int b, int c) : Calculate;
Calculate objCal = new Multiply(2, 3, 4);
switch (objCal)
{
    case Add(int a, int b, int c):
        //code goes here
        break;
    case Substract(int a, int b):
        //code goes here
        break;
    case Multiply(int a, int b, int c):
        //code goes here
        break;
    default:
        //default case
        break;
}
```
#### 通过Ref返回
```cs
ref string getFromList(string strVal, string[] Values)
{
 foreach (string val1 in Values)
 {
     if (strVal == val1)
        return ref val1; //return location as ref not actual value
 }
}
string[] values = { "a", "b", "c", "d" };

ref string strSubstitute = ref getFromList("b", values);

strSubstitute = "K"; // replaces 7 with 9 in the array

System.Write(values[1]); // it prints "K"
```
####  表达式错误
Throw Exception from Expression
```cs
public string getEmpInfo( string EmpName)
{
    string[] empArr = EmpName.Split(",");
    return (empArr.Length > 0) ? empArr[0] : throw new Exception("Emp Info Not exist");
}
```		
