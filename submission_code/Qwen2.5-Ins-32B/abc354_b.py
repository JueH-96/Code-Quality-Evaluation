# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1::2]
    C = list(map(int, data[2::2]))
    
    # Sort users by their names in lexicographical order
    sorted_users = sorted(zip(S, C))
    
    # Calculate the sum of the ratings
    T = sum(C)
    
    # Determine the winner based on the modulo operation
    winner_index = T % N
    winner_name = sorted_users[winner_index][0]
    
    print(winner_name)

if __name__ == "__main__":
    main()