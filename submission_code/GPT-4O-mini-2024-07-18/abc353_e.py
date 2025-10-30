def longest_common_prefix(x, y):
    min_length = min(len(x), len(y))
    for i in range(min_length):
        if x[i] != y[i]:
            return i
    return min_length

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    strings = data[1].split()
    
    total_sum = 0
    
    # Calculate the sum of longest common prefixes
    for i in range(N):
        for j in range(i + 1, N):
            total_sum += longest_common_prefix(strings[i], strings[j])
    
    print(total_sum)

if __name__ == "__main__":
    main()