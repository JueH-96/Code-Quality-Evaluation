def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    scores = list(map(int, input_data[1:]))
    
    # If there is only one person, no extra points are needed.
    if N == 1:
        print(0)
        return

    # Find the maximum score amongst the other people.
    max_other = max(scores[1:])
    
    # If person 1's score is already strictly greater than all others, no extra points are needed.
    if scores[0] > max_other:
        print(0)
    else:
        # Otherwise, calculate the extra points needed.
        print(max_other - scores[0] + 1)

if __name__ == '__main__':
    main()