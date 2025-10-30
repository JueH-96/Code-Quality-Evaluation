def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast parsing of inputs
    # First line: N, Q
    N, Q = map(int, input_data[:2])
    idx = 2
    
    # Next N integers: initial colors C_i
    colors = list(map(int, input_data[idx:idx+N]))
    idx += N
    
    # Prepare data structures:
    # colorCount[i] will be a dictionary {color: frequency} for box i
    # distinctCount[i] will track how many distinct colors are currently in box i
    colorCount = [dict() for _ in range(N)]
    distinctCount = [0]*N
    
    for i, c in enumerate(colors):
        colorCount[i][c] = 1
        distinctCount[i] = 1
    
    # Process queries
    out = []
    for _ in range(Q):
        a = int(input_data[idx]) - 1
        b = int(input_data[idx+1]) - 1
        idx += 2
        
        # If box a is empty, just print the distinct count of b
        if not colorCount[a]:
            out.append(str(distinctCount[b]))
            continue
        
        # Small-to-large merging strategy
        if len(colorCount[a]) <= len(colorCount[b]):
            # Merge contents of a into b
            new_count = distinctCount[b]
            dictA = colorCount[a]
            dictB = colorCount[b]
            
            for color, freq in dictA.items():
                old_freq = dictB.get(color, 0)
                if old_freq == 0:
                    new_count += 1
                dictB[color] = old_freq + freq
            
            dictA.clear()
            distinctCount[a] = 0
            distinctCount[b] = new_count
            out.append(str(new_count))
        else:
            # Merge contents of b into a, then make a empty and move final dictionary to b
            new_count = distinctCount[a]
            dictA = colorCount[a]
            dictB = colorCount[b]
            
            for color, freq in dictB.items():
                old_freq = dictA.get(color, 0)
                if old_freq == 0:
                    new_count += 1
                dictA[color] = old_freq + freq
            
            dictB.clear()
            distinctCount[b] = new_count
            distinctCount[a] = 0
            colorCount[b] = dictA  # b now has the merged dictionary
            colorCount[a] = {}     # a becomes empty
            out.append(str(new_count))
    
    # Print all results
    print("
".join(out))

# Do not forget to call main()
main()