# Step 1

Download dataset from GitHub

``` 
git clone https://github.com/yungshenglu/USTC-TFC2016.git
```

# Step 2 (Generate Netflow logs)
### Sample command for reading pcap file and filtering packets based on parameters (e.g., IP)
```
tcpdump -r <file.pcap> -w <filtered-file.pcap> 'host 192.168.10.8'
```

### Generate nflow from pcap files.
```
nfpcapd -r <path_to_pcap_file> -l <output_directory>
```

### Export netflow with nfdump and store in csv format
```
cd <output_directory>
nfdump -R <nflow_files_directory> -B -o extended -o csv > <output_file>
```

P.S. check this gist [gist](https://gist.github.com/jjsantanna/f2ee2f1fe23208299f4a2ca392f8b23f?permalink_comment_id=3749338) for installation instructions and troubleshooting

# Step 3 Generate Zeek logs
### Generate conn.log, http.log, dns.log, etc.
```
zeek -r <pcap>
```

# Step 4 Merge Zeek and Netflow logs

Execute notebook *merge_USTC-TFC2016.ipynb*

**P.S be careful with the exact directories for pcap, csv, log and nfcapd files around the dataset directories.**