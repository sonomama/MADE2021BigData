The description of the hw can be found in the pdf.

I created the cluster locally using the [docker-hadoop](https://github.com/big-data-europe/docker-hadoop) repo
and running ```docker compose up --build```. 

The second part is in the commands.txt file.

I was not able to run mapreduce locally due to memory issues, so the third part was done on the remote cluster.
On the cluster the job can be run as ```source run.sh```.

The results that I got:  
<pre>
          mean: 152.7206871868289 |           variance:  57672.84569843359  
MapReduce mean: 152.7206871868289 | MapReduce variance:  57672.84569843358
</pre>
