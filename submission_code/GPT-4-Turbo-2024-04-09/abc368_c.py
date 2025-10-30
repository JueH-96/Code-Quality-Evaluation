import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    H = list(map(int, data[1:]))
    
    T = 0
    for health in H:
        # Calculate how many normal hits (damage 1) and triple hits (damage 3) are needed
        normal_hits = (health % 3) + (health // 3)
        triple_hits = health // 3
        
        # Total time increments for this enemy
        T += normal_hits + 2 * triple_hits
    
    print(T)

if __name__ == "__main__":
    main()