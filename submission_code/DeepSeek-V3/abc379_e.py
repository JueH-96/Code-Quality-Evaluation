# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    total = 0
    current_sum = 0
    for i in range(N):
        current_sum = current_sum * 10 + int(S[i]) * (i + 1)
        total += current_sum
    
    print(total)

if __name__ == "__main__":
    main()