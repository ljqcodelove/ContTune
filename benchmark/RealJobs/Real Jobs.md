# Real Jobs

In our Oceanus platform, there is a UI bug that the TPS of source is 0, because the TPS here is the input TPS. 
But in databases, we can get the both TPS of input and output.

![VideoStreaming](https://github.com/ljqcodelove/ContTune/raw/main/benchmark/RealJobs/72582_Videostreaming_488Cores.png)

A job for video streaming in Tencent Meeting.

Before tuning, it has configured 488 Cores. 



![ETL](https://github.com/ljqcodelove/ContTune/raw/main/benchmark/RealJobs/72590_ETL_350Cores.png)

A job for ETL to clickhouse and kafak for Wechat.

Before tuning, it has configured 350 Cores.



![Monitoring](https://github.com/ljqcodelove/ContTune/raw/main/benchmark/RealJobs/72642_Monitoring_72Cores.png)

A job for monitoring for CSIG. 

Before tuning, it has configured 72 Cores.





