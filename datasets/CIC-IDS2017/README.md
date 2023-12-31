# Step 1
Dataset description at [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)


Download dataset from [dataset](http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/PCAPs/)

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

Execute notebook *merge_CIC-IDS2017_day.ipynb*

**P.S be careful with the exact directories for pcap, csv, log and nfcapd files around the dataset directories.**
