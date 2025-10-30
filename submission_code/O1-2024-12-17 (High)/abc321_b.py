def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    X = int(data[1])
    A = [int(x) for x in data[2:]]
    
    # N-1 scores are known (A). We want the minimum integer "candidate"
    # between 0 and 100 inclusive such that inserting "candidate" into A
    # and ignoring the lowest and highest scores yields a sum >= X.
    
    answer = -1
    for candidate in range(101):
        scores = A + [candidate]
        scores.sort()
        # Remove the lowest and the highest score
        mid_scores = scores[1:-1]
        if sum(mid_scores) >= X:
            answer = candidate
            break
    
    print(answer)

# Do not forget to call main()
if __name__ == "__main__":
    main()