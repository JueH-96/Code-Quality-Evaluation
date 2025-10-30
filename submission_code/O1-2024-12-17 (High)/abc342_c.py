def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]
    Q = int(data[2])
    
    # domain_of[x] will hold a list of the 'source' letters (0..25) that currently map to x
    domain_of = [[] for _ in range(26)]
    
    # Initially, each letter maps to itself
    for i in range(26):
        domain_of[i].append(i)
    
    idx = 3
    for _ in range(Q):
        c = data[idx]
        d = data[idx+1]
        idx += 2
        src = ord(c) - ord('a')
        dst = ord(d) - ord('a')
        if src != dst:
            # Move all domain letters from src to dst
            domain_of[dst].extend(domain_of[src])
            domain_of[src].clear()
    
    # Build final mapping for each of the 26 letters
    final_map = [0]*26
    for letter in range(26):
        for origin in domain_of[letter]:
            final_map[origin] = letter
    
    # Apply final mapping to S
    result = []
    for ch in S:
        mapped = final_map[ord(ch) - ord('a')]
        result.append(chr(mapped + ord('a')))
    
    print("".join(result))

# Do not forget to call main() at the end
if __name__ == "__main__":
    main()