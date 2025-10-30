import sys
import math

def main():
    input_data = sys.stdin.read().strip()
    A, B = map(int, input_data.split())
    
    # Calculate the minimum number of attacks needed
    attacks = math.ceil(A / B)
    
    print(attacks)

if __name__ == "__main__":
    main()