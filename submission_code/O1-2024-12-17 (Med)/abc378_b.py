def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    idx = 1
    garbage_types = []
    for _ in range(N):
        q_i = int(data[idx])
        r_i = int(data[idx+1])
        idx += 2
        garbage_types.append((q_i, r_i))
    
    Q = int(data[idx])
    idx += 1
    
    for _ in range(Q):
        t_j = int(data[idx]); d_j = int(data[idx+1])
        idx += 2
        
        q, r = garbage_types[t_j - 1]
        day_mod = d_j % q
        diff = (r - day_mod) % q
        print(d_j + diff)

# Call main() at the end
if __name__ == "__main__":
    main()