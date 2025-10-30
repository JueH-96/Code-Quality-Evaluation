def longest_common_prefix(x, y):
    i = 0
    while i < len(x) and i < len(y) and x[i] == y[i]:
        i += 1
    return i

def main():
    N = int(input().strip())
    strings = input().strip().split()
    
    total = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            total += longest_common_prefix(strings[i], strings[j])
    
    print(total)

if __name__ == "__main__":
    main()