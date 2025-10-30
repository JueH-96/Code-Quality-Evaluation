def main():
    import sys
    data = sys.stdin.read().strip().split()
    S, T = data[0], data[1]
    
    lenS = len(S)
    # We look for 1 <= c <= w < |S|
    for w in range(1, lenS):  # w goes from 1 to lenS-1
        for c in range(1, w + 1):  # c goes from 1 to w
            # Generate chunks of length w
            chunks = [S[i:i+w] for i in range(0, lenS, w)]
            
            # Concatenate c-th character from each chunk (if it exists)
            res = []
            for chunk in chunks:
                if len(chunk) >= c:
                    res.append(chunk[c-1])
            
            # Check if the result equals T
            if "".join(res) == T:
                print("Yes")
                return
    
    print("No")

# Do not forget to call main()!
main()