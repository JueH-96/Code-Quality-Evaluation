def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, X = map(int, data[:2])
    scores = list(map(int, data[2:2+N]))
    result = sum(score for score in scores if score <= X)
    print(result)

# Do not remove the following line
main()