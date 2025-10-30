def main():
    import sys
    input = sys.stdin.readline
    N = int(input())
    P = list(map(int, input().split()))
    for pi in P:
        # rank is 1 plus the number of people with a strictly higher score
        print(1 + sum(1 for x in P if x > pi))

# call main to execute the program
main()