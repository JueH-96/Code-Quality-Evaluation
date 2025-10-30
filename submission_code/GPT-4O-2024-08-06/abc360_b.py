# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    S = data[0]
    T = data[1]
    
    len_s = len(S)
    len_t = len(T)
    
    for w in range(1, len_s):
        for c in range(1, w + 1):
            # Collect the c-th characters from each valid chunk
            constructed_t = []
            for i in range(0, len_s, w):
                if i + c <= len_s:
                    constructed_t.append(S[i + c - 1])
            # Join the collected characters
            if ''.join(constructed_t) == T:
                print("Yes")
                return
    
    print("No")

if __name__ == "__main__":
    main()