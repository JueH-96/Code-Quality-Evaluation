# YOUR CODE HERE
def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    c1 = data[1]
    c2 = data[2]
    S = data[3]

    answer = ''.join(char if char == c1 else c2 for char in S)
    print(answer)

main()