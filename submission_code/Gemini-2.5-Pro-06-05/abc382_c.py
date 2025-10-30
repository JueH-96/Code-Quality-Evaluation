import sys
import heapq

def solve():
    """
    Reads input, solves the Conveyor Belt Sushi problem, and prints the output.
    """
    # Use fast I/O
    input = sys.stdin.readline

    try:
        # Read problem size
        n, m = map(int, input().split())
        
        # Read person gourmet levels and sushi deliciousness
        a = list(map(int, input().split()))
        b = list(map(int, input().split()))
    except (IOError, ValueError):
        # Handles potential empty lines at the end of input files
        return

    # Create tuples (value, original_index) to keep track of original positions after sorting.
    # People are 1-indexed in the output.
    people_sorted = sorted([(a[i], i + 1) for i in range(n)])
    sushi_sorted = sorted([(b[j], j) for j in range(m)])

    # `answers` will store the result for each sushi in its original input order.
    answers = [-1] * m
    
    # `available_people_heap` is a min-heap storing the indices of people who are
    # qualified to eat the current sushi. A person `i` is qualified for a sushi with
    # deliciousness `d` if `A_i <= d`. The heap allows finding the qualified
    # person with the minimum index in O(1) time.
    available_people_heap = []
    
    # A pointer to iterate through the sorted list of people.
    person_ptr = 0

    # Iterate through the sorted sushi (the "sweep-line" moves across deliciousness values).
    for sushi_deliciousness, sushi_original_index in sushi_sorted:
        
        # Add all people whose gourmet level is low enough for the current sushi
        # to the pool of available eaters.
        while person_ptr < n and people_sorted[person_ptr][0] <= sushi_deliciousness:
            # This person is a candidate. Add their original index to the min-heap.
            _, person_index = people_sorted[person_ptr]
            heapq.heappush(available_people_heap, person_index)
            person_ptr += 1
            
        # For the current sushi, the person with the smallest index among all qualified
        # candidates will eat it. This corresponds to the minimum element in our heap.
        if available_people_heap:
            # We peek at the heap's minimum element (`heap[0]`). We do not pop because
            # a person can eat multiple pieces of sushi; their availability resets for each new piece.
            eater_index = available_people_heap[0]
            answers[sushi_original_index] = eater_index
        # If the heap is empty, no one can eat this sushi, so the answer remains -1.

    # Print the final answers in the original order.
    for ans in answers:
        print(ans)

if __name__ == "__main__":
    solve()