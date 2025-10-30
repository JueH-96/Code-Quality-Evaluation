def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]
    
    # We want to find the first position (1-based index) where "ABC" appears.
    for i in range(N - 2):
        if S[i:i+3] == "ABC":
            print(i + 1)  # 1-based index
            return
    
    # If no occurrence found
    print(-1)

# Do not forget to call main()
main()