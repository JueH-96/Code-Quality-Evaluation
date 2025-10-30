def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    ptr = 0
    n = int(data[ptr])
    ptr += 1
    
    logs = []
    for _ in range(n):
        server = int(data[ptr])
        time = int(data[ptr + 1])
        logs.append((time, server))
        ptr += 2
    
    logs.sort()
    
    x = int(data[ptr])
    ptr += 1
    
    q_count = int(data[ptr])
    ptr += 1
    
    queries = []
    for idx in range(q_count):
        q = int(data[ptr])
        ptr += 1
        queries.append((q, idx))
    
    queries.sort()
    
    active_servers = set()
    result = [0] * q_count
    i = 0
    
    for q, idx in queries:
        q_minus_x = q - x
        # Add logs up to q
        while i < len(logs) and logs[i][0] <= q:
            server = logs[i][1]
            if server not in active_servers:
                active_servers.add(server)
            i += 1
        # Remove logs before q_minus_x
        while i > 0 and logs[i-1][0] < q_minus_x:
            server = logs[i-1][1]
            if server in active_servers:
                active_servers.remove(server)
            i -= 1
        result[idx] = len(active_servers)
    
    for num in result:
        print(num)

if __name__ == "__main__":
    main()