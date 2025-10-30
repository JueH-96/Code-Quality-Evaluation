import sys
from fractions import Fraction

def main():
    N = int(sys.stdin.readline())
    
    people = [] # List to store tuples: (original_id, A_heads, B_tails)
    
    for i in range(1, N + 1):
        # Read A_i and B_i for person i
        line = sys.stdin.readline().split()
        A = int(line[0])
        B = int(line[1])
        # Store as a tuple: (original_id, A_heads, B_tails)
        people.append((i, A, B))
        
    # Sort the list of people.
    # Each element p in 'people' is a tuple (original_id, A_heads, B_tails).
    # p[0] refers to original_id
    # p[1] refers to A_heads
    # p[2] refers to B_tails
    
    # The success rate is p[1] / (p[1] + p[2]).
    # The problem guarantees A_i + B_i >= 1, so p[1] + p[2] is always at least 1.
    
    # We need to sort in descending order of success rates.
    # Ties are broken by ascending order of original_id.
    
    # The key for sorting will be a tuple: (-success_rate, original_id).
    # Python's default sort is ascending. By negating the success_rate,
    # we achieve descending order for it. The original_id is kept positive
    # for ascending order tie-breaking.
    
    # We use `fractions.Fraction` for precise calculation of success_rate.
    # The lambda function defines how to extract the key from an element p.
    # key_function = lambda p: (-Fraction(p[1], p[1] + p[2]), p[0])
    
    people.sort(key=lambda p: (-Fraction(p[1], p[1] + p[2]), p[0]))
    
    # After sorting, 'people' contains the tuples in the desired order.
    # We need to print the original_ids (which is the first element of each tuple).
    sorted_ids = [p[0] for p in people]
    
    # Print the sorted ids, space-separated, followed by a newline.
    sys.stdout.write(" ".join(map(str, sorted_ids)) + "
")

if __name__ == '__main__':
    main()