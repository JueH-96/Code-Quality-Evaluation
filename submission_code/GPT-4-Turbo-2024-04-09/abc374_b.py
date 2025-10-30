# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    S = data[0]
    T = data[1]
    
    len_s = len(S)
    len_t = len(T)
    min_len = min(len_s, len_t)
    
    for i in range(min_len):
        if S[i] != T[i]:
            print(i + 1)
            return
    
    if len_s == len_t:
        print(0)
    else:
        print(min_len + 1)

if __name__ == "__main__":
    main()