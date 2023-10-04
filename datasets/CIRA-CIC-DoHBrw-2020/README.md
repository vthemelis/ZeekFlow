# Step 1
Dataset description at [CIRA-CIC-DoHBrw-2020](https://www.unb.ca/cic/datasets/dohbrw-2020.html)


Download dataset from [dataset](http://205.174.165.80/CICDataset/DoHBrw-2020/Dataset/)

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

**P.S #2 for ZeekFlow we utilized: Malicious (MaliciousDoH-dns2tcp-Pcap-001_600:) and Benign (Google pcap: dump_00001_20200113100617.pcap, Cloudflare pcap:dump_00001_20200113152847.pcap, Quad9: dump_00001_20200111222621.pcap)**
