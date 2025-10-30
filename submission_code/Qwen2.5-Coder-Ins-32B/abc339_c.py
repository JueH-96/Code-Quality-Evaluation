# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:]))
    
    current_passengers = 0
    min_passengers = 0
    
    for a in A:
        current_passengers += a
        if current_passengers < min_passengers:
            min_passengers = current_passengers
    
    initial_passengers = -min_passengers
    final_passengers = initial_passengers + sum(A)
    
    print(final_passengers)

if __name__ == "__main__":
    main()