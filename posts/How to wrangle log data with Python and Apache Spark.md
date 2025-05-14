---
title: "How to wrangle log data with Python and Apache Spark"
source: "https://opensource.com/article/19/5/log-data-apache-spark"
author:
  - "[[Dipanjan (DJ) Sarkar]]"
published:
created: 2025-02-07
description: "Case study with NASA logs to show how Spark can be leveraged for analyzing data at scale."
tags:
  - "clippings"
---
One of the most popular and effective enterprise use-cases which leverage analytics today is log analytics. Nearly every organization today has multiple systems and infrastructure running day in and day out. To effectively keep their business running, these organizations need to know if their infrastructure is performing to its maximum potential. Finding out involves analyzing system and application logs and maybe even applying predictive analytics on log data. The amount of log data involved is typically massive, depending on the type of organizational infrastructure involved and applications running on it. 

![The log data processing pipeline by Doug Henschen.](https://opensource.com/sites/default/files/uploads/processing_and_wrangling_nasa_logs_figure_1_500.jpg "The log data processing pipeline")

Image by:

<sup>The log data processing pipeline</sup>

Gone are the days when we were limited to analyzing a data sample on a single machine due to compute constraints. Powered by big data, better and distributed computing, and frameworks like [Apache Spark](https://spark.apache.org/) for big data processing and open source analytics, we can perform scalable log analytics on potentially billions of log messages daily. The intent of this case study-oriented tutorial is to take a hands-on approach showcasing how we can leverage Spark to perform log analytics at scale on semi-structured log data. If you are interested in scalable [SQL](https://en.wikipedia.org/wiki/SQL) with Spark, feel free to check out [*SQL at scale with Spark*](https://towardsdatascience.com/sql-at-scale-with-apache-spark-sql-and-dataframes-concepts-architecture-and-examples-c567853a702f).

While there are many excellent open source frameworks and tools out there for log analytics—such as [Elasticsearch](https://www.elastic.co/)—the intent of this two-part tutorial is to showcase how Spark can be leveraged for analyzing logs at scale. In the real world, you are of course free to choose your own toolbox when analyzing your log data.

Let’s get started!

## Main objective: NASA log analytics

As we mentioned before, Apache Spark is an excellent and ideal open source framework for wrangling, analyzing and modeling structured and unstructured data—at scale! In this tutorial, our main objective is one of the most popular use-cases in the industry—log analytics. Server logs are a common enterprise data source and often contain a gold mine of actionable insights and information. Log data comes from many sources in these conditions, such as the web, client and compute servers, applications, user-generated content, and flat files. These logs can be used for monitoring servers, improving business and customer intelligence, building recommendation systems, fraud detection, and much more.

Spark allows you to cheaply dump and store your logs into files on disk, while still providing rich APIs to perform data analysis at scale. This hands-on case study will show you how to use Apache Spark on real-world production logs from [NASA](https://www.nasa.gov/) while learning data wrangling and basic yet powerful techniques for exploratory data analysis. In this study, we will analyze log datasets from the [NASA Kennedy Space Center](https://www.nasa.gov/centers/kennedy/home/index.html) web server in Florida.

The full data set—containing two months’ worth of all HTTP requests to the NASA Kennedy Space Center—is freely available [here](http://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html) for download. Or, if you prefer FTP:

- Jul 01 to Jul 31, ASCII format, 20.7 MB gzip compressed, 205.2 MB uncompressed: [ftp://ita.ee.lbl.gov/traces/NASA\_access\_log\_Jul95.gz](https://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz)
- Aug 04 to Aug 31, ASCII format, 21.8 MB gzip compressed, 167.8 MB uncompressed: [ftp://ita.ee.lbl.gov/traces/NASA\_access\_log\_Aug95.gz](https://ita.ee.lbl.gov/traces/NASA_access_log_Aug95.gz)

Next, if you want to follow along, download the tutorial from [***my GitHub***](https://github.com/dipanjanS/data_science_for_all/tree/master/tds_scalable_log_analytics) and place both of these files in the same directory as the tutorial’s [Jupyter Notebook](https://jupyter.org/).

## Setting up dependencies

The first step is to make sure you have access to a Spark session and cluster. For this step, you can use your own local Spark setup or a cloud-based setup. Typically, most cloud platforms provide a Spark cluster these days and you also have free options, including [Databricks community edition](https://community.cloud.databricks.com/). This tutorial assumes you already have Spark set up, hence we will not be spending additional time configuring or setting up Spark from scratch.

Often pre-configured Spark setups already have the necessary environment variables or dependencies pre-loaded when you start your Jupyter Notebook server. In my case, I can check them using the following commands in my notebook:

```python
spark
```

![Spark session](https://opensource.com/sites/default/files/uploads/spark_session.png "Spark session")

These results show me that my cluster is running Spark 2.4.0 at the moment. We can also check if **sqlContext** is present using the following code:

```python
sqlContext
```

`**<pyspark.sql.context.SQLContext at 0x7fb1577b6400>**`

Now in case you don’t have these variables pre-configured and get an error, you can load and configure them using the following code:

```python
# configure spark variables
from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.sql.session import SparkSession
    
sc = SparkContext()
sqlContext = SQLContext(sc)
spark = SparkSession(sc)

# load up other dependencies
import re
import pandas as pd
```

We also need to load other libraries for working with [DataFrames](https://pandas.pydata.org/pandas-docs/stable/getting_started/dsintro.html#dataframe) and [regular expressions](https://en.wikipedia.org/wiki/Regular_expression). Working with regular expressions is one of the major aspects of parsing log files. This tool offers a powerful pattern-matching technique which can be used to extract and find patterns in semi-structured and unstructured data.

![The Perl Problems strip from xkcd.](https://opensource.com/sites/default/files/perl_problems_600.png "The Perl Problems strip from xkcd.")

Image by:

<sup>The Perl Problems strip from <a href="https://www.xkcd.com/1171/" target="_blank" rel="ugc">xkcd</a>.</sup>

Regular expressions can be extremely effective and powerful, yet they can also be overwhelming and confusing. Not to worry though, with practice you can really leverage their maximum potential. The following example showcases a way of using regular expressions in [Python](https://www.python.org/). Here, we try to find all occurences of the word *'spark'* in a given input sentence.

```python
m = re.finditer(r'.*?(spark).*?', "I'm searching for a spark in PySpark", re.I)
for match in m:
    print(match, match.start(), match.end())
```

`**<_sre.SRE_Match object; span=(0, 25), match=“I’m searching for a spark”> 0 25**`

`** <_sre.SRE_Match object; span=(25, 36), match=’ in PySpark’> 25 36**`

Let’s move on to the next part of our analysis.

## Loading and Viewing the NASA Log Dataset

Given that our data is stored in the following path (in the form of flat files), let’s load it into a DataFrame. We’ll do this in steps. The following code loads our disk’s log data file names:

```python
import glob

raw_data_files = glob.glob('*.gz')
raw_data_files
```

`**[‘NASA_access_log_Jul95.gz’, ‘NASA_access_log_Aug95.gz’]**`

Now, we’ll use **sqlContext.read.text()** or **spark.read.text()** to read the text file. This code produces a DataFrame with a single string column called **value**:

```text
base_df = spark.read.text(raw_data_files)
base_df.printSchema()
```

`**root**`

`** |-- value: string (nullable = true)**`

This output allows us to see the text for our log data’s schema that we will soon inspect. You can view the type of data structure holding our log data using the following code:

```python
type(base_df)
```

`**pyspark.sql.dataframe.DataFrame**`

Throughout this tutorial we use Spark DataFrames. However if you want, you can also convert a DataFrame into a [Resilient Distributed Dataset (RDD)](https://en.wikipedia.org/wiki/Apache_Spark)—Spark’s original data structure ()—if needed by adding the following code:

```python
base_df_rdd = base_df.rdd
type(base_df_rdd)
```

`**pyspark.rdd.RDD**`

Let’s now take a peek at the actual log data in our DataFrame: 

```python
base_df.show(10, truncate=False)
```

![The log data within the dataframe.](https://opensource.com/sites/default/files/uploads/processing_and_wrangling_nasa_logs_figure_4_600_0.png "The log data within the base_df.show dataframe")

Image by:

The log data within the base\_df.show dataframe

This result definitely looks like standard semi-structured server log data. We will definitely need to do some data processing and wrangling before this file is useful. Do remember that accessing data from RDDs is slightly different as seen below:

```python
base_df_rdd.take(10)
```

![Log data from the resilient distributed datasets.](https://opensource.com/sites/default/files/uploads/processing_and_wrangling_nasa_logs_figure_5_600_0.png "Figure 5: The log data within the dataframe via base_df_rdd")

Image by:

Figure 5: The log data within the dataframe via base\_df\_rdd

Now that we have loaded and viewed our log data, let’s process and wrangle it.

## Data Wrangling

In this section, we clean and parse our log dataset to extract structured attributes with meaningful information from each log message.

### Log data understanding

If you’re familiar with web server logs, you’ll recognize that the data displayed above is in [Common Log Format](https://www.w3.org/Daemon/User/Config/Logging.html#common-logfile-format). The fields are:

`**remotehost**** rfc931 ****authuser**** [date] "request" status bytes**`

| Field | Description |
| --- | --- |
| *remotehost* | Remote hostname (or IP number if DNS hostname is not available or if [DNSLookup](https://www.w3.org/Daemon/User/Config/General.html#DNSLookup) is off). |
| *rfc931* | The remote logname of the user if at all it is present. |
| *authuser* | The username of the remote user after authentication by the HTTP server. |
| *\[date\]* | Date and time of the request. |
| *“request”* | The request, exactly as it came from the browser or client. |
| *status* | The [HTTP status code](https://en.wikipedia.org/wiki/List_of_HTTP_status_codes) the server sent back to the client. |
| *bytes* | The number of bytes (`Content-Length`) transferred to the client. |

We now need techniques to parse, match, and extract these attributes from the log data.

### Data parsing and extraction with regular expressions

Next, we have to parse our semi-structured log data into individual columns. We’ll use the special built-in **[regexp\_extract()](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.regexp_extract)** function to do the parsing. This function matches a column against a regular expression with one or more **[capture](http://regexone.com/lesson/capturing_groups)** [**groups**](http://regexone.com/lesson/capturing_groups), and allows you to extract one of the matched groups. We’ll use one regular expression for each field we wish to extract.

You must have heard or used a fair bit of regular expressions by now. If you find regular expressions confusing (and they certainly *can* be), and you want to learn more about them, we recommend checking out the [RegexOne web site](http://regexone.com/). You might also find [*Regular Expressions Cookbook*](http://shop.oreilly.com/product/0636920023630.do), by Goyvaerts and Levithan, to be a useful reference.

Let’s take a look at the total number of logs we are working within our dataset:

```python
print((base_df.count(), len(base_df.columns)))
```

`**(3461613, 1)**`

Looks like we have a total of approximately 3.46 million log messages. Not a small number! Let’s extract and take a look at some sample log messages:

```python
sample_logs = [item['value'] for item in base_df.take(15)]
sample_logs
```

#### 

![Sample log messages.](https://opensource.com/sites/default/files/uploads/processing_and_wrangling_nasa_logs_figure_6_600.png "Sample log messages.")

Image by:

<sup>Sample log messages.</sup>

Let’s write some regular expressions to extract the hostname from the logs:

```python
host_pattern = r'(^\S+\.[\S+\.]+\S+)\s'
hosts = [re.search(host_pattern, item).group(1)
           if re.search(host_pattern, item)
           else 'no match'
           for item in sample_logs]
hosts
```

`**[‘199.72.81.55’,**`

`** ‘unicomp6.unicomp.net’,**`

`** ‘199.120.110.21’,**`

`** ‘burger.letters.com’,**`

`** …,**`

`** …,**`

`** ‘unicomp6.unicomp.net’,**`

`** ‘d104.aa.net’,**`

`** ‘d104.aa.net’]**`

Let’s use regular expressions to extract the timestamp fields from the logs:

```python
ts_pattern = r'\[(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} -\d{4})]'
timestamps = [re.search(ts_pattern, item).group(1) for item in sample_logs]
timestamps
```

`**[‘01/Jul/1995:00:00:01 -0400’,**`

`** ‘01/Jul/1995:00:00:06 -0400’,**`

`** ‘01/Jul/1995:00:00:09 -0400’,**`

`** …,**`

`** …,**`

`** ‘01/Jul/1995:00:00:14 -0400’,**`

`** ‘01/Jul/1995:00:00:15 -0400’,**`

`** ‘01/Jul/1995:00:00:15 -0400’]**`

### Extracting HTTP request method, URIs, and protocol

Let’s now use regular expressions to extract the HTTP request methods, [URIs](https://en.wikipedia.org/wiki/Uniform_Resource_Identifier), and Protocol patterns fields from the logs:

```python
method_uri_protocol_pattern = r'\"(\S+)\s(\S+)\s*(\S*)\"'
method_uri_protocol = [re.search(method_uri_protocol_pattern, item).groups()
               if re.search(method_uri_protocol_pattern, item)
               else 'no match'
              for item in sample_logs]
method_uri_protocol
```

`**[(‘GET’, ‘/history/apollo/’, ‘HTTP/1.0’),**`

`** (‘GET’, ‘/shuttle/countdown/’, ‘HTTP/1.0’),**`

`** …,**`

`** …,**`

`** (‘GET’, ‘/shuttle/countdown/count.gif’, ‘HTTP/1.0’),**`

`** (‘GET’, ‘/images/NASA-logosmall.gif’, ‘HTTP/1.0’)]**`

Let’s now use regular expressions to extract the HTTP status codes from the logs: 

```python
status_pattern = r'\s(\d{3})\s'
status = [re.search(status_pattern, item).group(1) for item in sample_logs]
print(status)
```

`**[‘200’, ‘200’, ‘200’, ‘304’, …, ‘200’, ‘200’]**`

### Extracting HTTP response content size

Let’s now use regular expressions to extract the HTTP response content size from the logs: 

```python
content_size_pattern = r'\s(\d+)$'
content_size = [re.search(content_size_pattern, item).group(1) for item in sample_logs]
print(content_size)
```

`**[‘6245’, ‘3985’, ‘4085’, ‘0’, …, ‘1204’, ‘40310’, ‘786’]**`

### Putting it all together

Let’s now leverage all the regular expression patterns we previously built and use the **`regexp_extract(...)`** method to build our DataFrame with all of the log attributes neatly extracted in their own separate columns.

```python
from pyspark.sql.functions import regexp_extract

logs_df = base_df.select(regexp_extract('value', host_pattern, 1).alias('host'),
                         regexp_extract('value', ts_pattern, 1).alias('timestamp'),
                         regexp_extract('value', method_uri_protocol_pattern, 1).alias('method'),
                         regexp_extract('value', method_uri_protocol_pattern, 2).alias('endpoint'),
                         regexp_extract('value', method_uri_protocol_pattern, 3).alias('protocol'),
                         regexp_extract('value', status_pattern, 1).cast('integer').alias('status'),
                         regexp_extract('value', content_size_pattern, 1).cast('integer').alias('content_size'))
logs_df.show(10, truncate=True)
print((logs_df.count(), len(logs_df.columns)))
```

![Log dataframe extracted using regexp_extract(...)](https://opensource.com/sites/default/files/uploads/processing_and_wrangling_nasa_logs_figure_7_600.png "Log dataframe extracted using regexp_extract(...)")

Image by:

<sup>Log dataframe extracted using regexp_extract(...)</sup>

### Finding missing values

Missing and null values are the bane of data analysis and machine learning. Let’s see how well our data parsing and extraction logic worked. First, let’s verify that there are no null rows in the original DataFrame:

```python
(base_df
    .filter(base_df['value']
                .isNull())
    .count())
```

`**0**`

All good! Now, if our data parsing and extraction worked properly, we should not have any rows with potential null values. Let’s try and put that to test:

```python
bad_rows_df = logs_df.filter(logs_df['host'].isNull()| 
                             logs_df['timestamp'].isNull() | 
                             logs_df['method'].isNull() |
                             logs_df['endpoint'].isNull() |
                             logs_df['status'].isNull() |
                             logs_df['content_size'].isNull()|
                             logs_df['protocol'].isNull())
bad_rows_df.count()
```

`**33905**`

Ouch! Looks like we have over 33K missing values in our data! Can we handle this?

Do remember, this is not a regular pandas (link) DataFrame which you can directly query and get which columns have null. Our so-called *big dataset* is residing on disk which can potentially be present in multiple nodes in a spark cluster. So how do we find out which columns have potential nulls?

### Finding null counts

We can typically use the following technique to find out which columns have null values.

**Note:** This approach is adapted from an [excellent answer](http://stackoverflow.com/a/33901312) on [StackOverflow](https://stackoverflow.com/).

```python
from pyspark.sql.functions import col
from pyspark.sql.functions import sum as spark_sum

def count_null(col_name):
    return spark_sum(col(col_name).isNull().cast('integer')).alias(col_name)

# Build up a list of column expressions, one per column.
exprs = [count_null(col_name) for col_name in logs_df.columns]

# Run the aggregation. The *exprs converts the list of expressions into
# variable function arguments.
logs_df.agg(*exprs).show()
```

![Checking which columns have null values.](https://opensource.com/sites/default/files/uploads/processing_and_wrangling_nasa_logs_figure_8.png "Checking which columns have null values with count_null()")

Image by:

<sup>Checking which columns have null values with count_null()</sup>

Well, looks like we have one missing value in the **status** column and everything else is in the **content\_size** column. Let’s see if we can figure out what’s wrong!

### Handling nulls in HTTP status

Our original parsing regular expression for the **status** column was:

```python
regexp_extract('value', r'\s(\d{3})\s', 1).cast('integer')
                                          .alias( 'status')
```

Could it be that there are more digits making our regular expression wrong? Or is the data point itself bad? Let’s find out.

**Note**: In the expression below, the tilde (`**~**)` means “not”.

```python
null_status_df = base_df.filter(~base_df['value'].rlike(r'\s(\d{3})\s'))
null_status_df.count()
```

`**1**`

Let’s look at what this bad record looks like:

```text
null_status_df.show(truncate=False)
```

![A bad record that's missing information.](https://opensource.com/sites/default/files/uploads/processing_and_wrangling_nasa_logs_figure_9.png "A bad record that's missing information via null_status_df")

Image by:

<sup>A bad record that's missing information via null_status_df</sup>

Looks like a record with a lot of missing information. Let’s pass this through our log data parsing pipeline:

```python
bad_status_df = null_status_df.select(regexp_extract('value', host_pattern, 1).alias('host'),
                                      regexp_extract('value', ts_pattern, 1).alias('timestamp'),
                                      regexp_extract('value', method_uri_protocol_pattern, 1).alias('method'),
                                      regexp_extract('value', method_uri_protocol_pattern, 2).alias('endpoint'),
                                      regexp_extract('value', method_uri_protocol_pattern, 3).alias('protocol'),
                                      regexp_extract('value', status_pattern, 1).cast('integer').alias('status'),
                                      regexp_extract('value', content_size_pattern, 1).cast('integer').alias('content_size'))
bad_status_df.show(truncate=False)
```

![The full bad log record containing no information and two null entries.](https://opensource.com/sites/default/files/uploads/processing_and_wrangling_nasa_logs_figure_10.png "The full bad log record containing no information and two null entries.")

Image by:

<sup>The full bad log record containing no information and two null entries.</sup>

Looks like the record itself is an incomplete record with no useful information, the best option would be to drop this record as follows:

```python
logs_df = logs_df[logs_df['status'].isNotNull()]
exprs = [count_null(col_name) for col_name in logs_df.columns]
logs_df.agg(*exprs).show()
```

![The dropped record.](https://opensource.com/sites/default/files/uploads/processing_and_wrangling_nasa_logs_figure_11.png "The dropped record.")

Image by:

<sup>The dropped record.</sup>

### Handling nulls in HTTP content size

Based on our previous regular expression, our original parsing regular expression for the **content\_size** column was:

```python
regexp_extract('value', r'\s(\d+)$', 1).cast('integer')
                                       .alias('content_size')
```

Could there be missing data in our original dataset itself? Let’s find out. We first find the records with potential missing content sizes in our base DataFrame:

```python
null_content_size_df = base_df.filter(~base_df['value'].rlike(r'\s\d+$'))
null_content_size_df.count()
```

`**33905**`

The number seems to match the number of missing content size values in our processed DataFrame. Let’s take a look at the top ten records of our data frame having missing content sizes:

```python
null_content_size_df.take(10)
```

![The top 10 dataframe records with missing content sizes.](https://opensource.com/sites/default/files/uploads/processing_and_wrangling_nasa_logs_figure_12_600.png "The top 10 dataframe records with missing content sizes.")

Image by:

<sub>The top 10 dataframe records with missing content sizes.</sub>

It is quite evident that the bad raw data records correspond to error responses, where no content was sent back and the server emitted a **`-`** for the `**content_size**` field. Since we don’t want to discard those rows from our analysis, let’s impute or fill them with 0.

### Fix the rows with null content\_size

The easiest solution is to replace the null values in `**logs_df**` with 0 like we discussed earlier. The Spark DataFrame API provides a set of functions and fields specifically designed for working with null values, among them:

- `[**fillna**](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.fillna)[**()**](https://opensource.com/%28http%3A//spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.fillna\))`, which fills null values with specified non-null values.
- `[**na**](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.na)`, which returns a **`[DataFrameNaFunctions](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrameNaFunctions)`** object with many functions for operating on null columns.

There are several ways to invoke this function. The easiest is just to replace all null columns with known values. But, for safety, it’s better to pass a Python dictionary containing `**(column_name, value)**` mappings. That’s what we’ll do. An example from the documentation is depicted below: 

```python
>>> df4.na.fill({'age': 50, 'name': 'unknown'}).show()
+---+------+-------+
|age|height|   name|
+---+------+-------+
| 10|    80|  Alice|
|  5|  null|    Bob|
| 50|  null|    Tom|
| 50|  null|unknown|
+---+------+-------+
```

Now we use this function to fill all the missing values in the `**content_size**` field with 0:

```python
logs_df = logs_df.na.fill({'content_size': 0})
exprs = [count_null(col_name) for col_name in logs_df.columns]
logs_df.agg(*exprs).show()
```

![The null values now replaced by zero.](https://opensource.com/sites/default/files/uploads/processing_and_wrangling_nasa_logs_figure_13.png "The null values now replaced by zero.")

Image by:

<sup>The null values now replaced by zero.</sup>

Look at that, no missing values!

### Handling temporal fields (timestamp)

Now that we have a clean, parsed DataFrame, we have to parse the timestamp field into an actual timestamp. The Common Log Format time is somewhat non-standard. A [User-Defined Function (UDF)](https://en.wikipedia.org/wiki/User-defined_function) is the most straightforward way to parse it: 

```python
from pyspark.sql.functions import udf

month_map = {
  'Jan': 1, 'Feb': 2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7,
  'Aug':8,  'Sep': 9, 'Oct':10, 'Nov': 11, 'Dec': 12
}

def parse_clf_time(text):
    """ Convert Common Log time format into a Python datetime object
    Args:
        text (str): date and time in Apache time format [dd/mmm/yyyy:hh:mm:ss (+/-)zzzz]
    Returns:
        a string suitable for passing to CAST('timestamp')
    """
    # NOTE: We're ignoring the time zones here, might need to be handled depending on the problem you are solving
    return "{0:04d}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}".format(
      int(text[7:11]),
      month_map[text[3:6]],
      int(text[0:2]),
      int(text[12:14]),
      int(text[15:17]),
      int(text[18:20])
    )
```

Let’s now use this function to parse our DataFrame's `**time** `column:

```python
udf_parse_time = udf(parse_clf_time)

logs_df = (logs_df.select('*', udf_parse_time(logs_df['timestamp'])
                                  .cast('timestamp')
                                  .alias('time'))
                  .drop('timestamp')
logs_df.show(10, truncate=True)
```

![The timestamp parsed with a User-Defined Function (UDF).](https://opensource.com/sites/default/files/uploads/processing_and_wrangling_nasa_logs_figure_14_600.png "The timestamp parsed with a User-Defined Function (UDF).")

Image by:

<sup>The timestamp parsed with a User-Defined Function (UDF).</sup>

Things seem to be looking good! Let’s verify this by checking our DataFrame's schema:

```python
logs_df.printSchema()
```

`**root**`

`** |-- host: string (nullable = true)**`

`** |-- method: string (nullable = true)**`

`** |-- endpoint: string (nullable = true)**`

`** |-- protocol: string (nullable = true)**`

`** |-- status: integer (nullable = true)**`

`** |-- content_size: integer (nullable = false)**`

` **|-- time: timestamp (nullable = true)**`

Let’s now cache  **`logs_df`** since we will be using it extensively for our data analysis section in [part two](https://opensource.com/article/19/5/visualizing-and-analyzing-nasa-logs) of this series.

```python
logs_df.cache()
```

## Conclusion

Acquiring, processing and wrangling data are some of the most important steps in any end-to-end Data Science or Analytics use-case. Things start getting more difficult when dealing with semi-structured or unstructured data at scale. This case study gives you a step-by-step hands-on approach to leveraging the power of open-source tools and frameworks like Python and Spark to process and wrangling semi-structured NASA log data at scale. Once we have prepared a clean dataset, we can finally start using it to gain useful insights about NASA servers. Click through to the second article in this series for a [hands-on tutorial](https://opensource.com/article/19/5/visualize-log-data-apache-spark) on Analyzing and Visualizing NASA log data with Python and Apache Spark.    

---

*This article originally appeared on Medium's [Towards Data Science](https://towardsdatascience.com/) channel and is republished with permission.* 

---