def main():
    import sys
    data = sys.stdin.read().split()
    S, T = data[0], data[1]
    
    n = len(S)
    # Try all possible w and c
    for w in range(1, n):        # 1 <= w < |S|
        for c in range(1, w+1):  # 1 <= c <= w
            chunks = [S[i:i+w] for i in range(0, n, w)]
            collected = []
            for chunk in chunks:
                if len(chunk) >= c:
                    collected.append(chunk[c-1])
            if ''.join(collected) == T:
                print("Yes")
                return
    print("No")

# Do not forget to call main()!
main()