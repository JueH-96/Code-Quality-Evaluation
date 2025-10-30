import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    Q = int(input[idx+1])
    idx += 2
    Ks = [int(input[idx+i]) for i in range(Q)]
    
    # Possible K values can be derived from the possible configurations
    # We need to find all K in the interval [0, N*N] where K can be written as the sum of the main diagonal matrices
    # According to the problem's specific conditions, the possible K values are generated based on quadratic residues
    # However, after research, I found that the solution involves checking K values of the form a*(2*N -a) where a ranges from 0 to N
    # This corresponds to considering submatrices formed by a rows and columns where the intersection cells are fixed
    
    possible_Ks = set()
    for a in range(0, N+1):
        possible_Ks.add(a * (2 * N - a))
    
    for k in Ks:
        print("Yes" if k in possible_Ks else "No")

if __name__ == "__main__":
    main()