def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    idx = 1
    out = []
    
    for _ in range(t):
        n = int(input_data[idx]); idx += 1
        k = int(input_data[idx]); idx += 1
        s = input_data[idx]; idx += 1
        
        covered_until = -1
        ops = 0
        
        for i in range(n):
            if i > covered_until and s[i] == 'B':
                ops += 1
                covered_until = i + k - 1
                if covered_until >= n:
                    break
        
        out.append(str(ops))
    
    print("
".join(out))

# do not forget to call main
if __name__ == "__main__":
    main()