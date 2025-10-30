def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast IO pointer
    idx = 0
    
    N = int(input_data[idx]); idx+=1
    Q = int(input_data[idx]); idx+=1
    
    # Read initial colors
    colors = list(map(int, input_data[idx:idx+N]))
    idx += N
    
    # freq[i] will be a dictionary (color -> count) for box i (1-based)
    freq = [dict() for _ in range(N+1)]
    for i in range(1, N+1):
        c = colors[i-1]
        freq[i][c] = freq[i].get(c, 0) + 1
    
    out = []
    
    # Process queries
    for _ in range(Q):
        a = int(input_data[idx]); idx+=1
        b = int(input_data[idx]); idx+=1
        
        # If freq[a] has fewer distinct colors than freq[b], merge a into b
        if len(freq[a]) <= len(freq[b]):
            # Move contents of freq[a] into freq[b]
            for color, cnt in freq[a].items():
                freq[b][color] = freq[b].get(color, 0) + cnt
            freq[a].clear()
            out.append(str(len(freq[b])))
        else:
            # Otherwise, merge b into a, then rename freq[a] into freq[b]
            for color, cnt in freq[b].items():
                freq[a][color] = freq[a].get(color, 0) + cnt
            # Now freq[a] holds everything, so point freq[b] to freq[a]
            bigger = freq[a]
            freq[a] = {}
            freq[b] = bigger
            out.append(str(len(freq[b])))
    
    print("
".join(out))

# Do not forget to call main()
if __name__ == "__main__":
    main()