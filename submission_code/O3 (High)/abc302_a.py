import sys

def main():
    # Read A and B
    A_str, B_str = sys.stdin.readline().split()
    A = int(A_str)
    B = int(B_str)
    
    # Minimum number of attacks = ceil(A / B)
    # Efficiently computed with integer arithmetic as:
    attacks = (A + B - 1) // B
    
    print(attacks)

if __name__ == "__main__":
    main()