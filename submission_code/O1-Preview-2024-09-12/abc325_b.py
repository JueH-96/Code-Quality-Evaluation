# YOUR CODE HERE
import sys

def main():
    import sys
    N = int(sys.stdin.readline())
    events = []
    for _ in range(N):
        W_i_str, X_i_str = sys.stdin.readline().split()
        W_i = int(W_i_str)
        X_i = int(X_i_str)

        start_i = (9 - X_i)%24
        end_i = (17 - X_i)%24

        if end_i >= start_i:
            # Interval does not wrap around
            events.append( (start_i, W_i) )
            events.append( (end_i, -W_i) )
            events.append( (start_i + 24, W_i) )  # For wrapping around after 24
            events.append( (end_i + 24, -W_i) )
        else:
            # Interval wraps around
            events.append( (start_i, W_i) )
            events.append( (end_i + 24, -W_i) )
    # Sort events by time
    events.sort()
    max_sum = 0
    curr_sum = 0
    for time, delta in events:
        curr_sum += delta
        if curr_sum > max_sum:
            max_sum = curr_sum
    print(max_sum)

if __name__ == "__main__":
    main()