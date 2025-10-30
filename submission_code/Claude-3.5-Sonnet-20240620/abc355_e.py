# YOUR CODE HERE
def query(i, j):
    print(f"? {i} {j}")
    response = int(input())
    if response == -1:
        exit()
    return response

def solve(N, L, R):
    result = 0
    current_length = 1
    current_start = L

    while current_start <= R:
        i = 0
        while (1 << (i + 1)) <= min(current_length, R - current_start + 1):
            i += 1

        j = current_start >> i
        response = query(i, j)
        
        segment_start = j << i
        segment_end = min((j + 1) << i, R + 1)
        segment_length = segment_end - segment_start
        
        if segment_start < L:
            left_overlap = L - segment_start
            response = (response - query(i, j - 1) + 100) % 100
            segment_length -= left_overlap

        if segment_end > R + 1:
            right_overlap = segment_end - (R + 1)
            response = (response - query(i - 1, (R + 1) >> (i - 1)) + 100) % 100
            segment_length -= right_overlap

        result = (result + response) % 100
        current_start = segment_end
        current_length = R - current_start + 1

    print(f"! {result}")

N, L, R = map(int, input().split())
solve(N, L, R)