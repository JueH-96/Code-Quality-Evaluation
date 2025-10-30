# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    H = list(map(int, data[1:]))
    
    T = 0
    for health in H:
        # Calculate the number of attacks needed when T is not a multiple of 3
        normal_attacks = health // 3
        remaining_health = health % 3
        
        # Calculate the total number of attacks needed
        if remaining_health == 0:
            T += 2 * normal_attacks
        elif remaining_health == 1:
            T += 2 * normal_attacks + 1
        else:  # remaining_health == 2
            T += 2 * normal_attacks + 2
    
    print(T)

if __name__ == "__main__":
    main()