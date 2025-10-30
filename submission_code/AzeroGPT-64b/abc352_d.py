import sys
input = sys.stdin.readline

def get_max_right_pos(arr, prev, high):
    low = 1
    high += 1
    while low < high:
        mid = (low+high)//2
        if all(arr[i] <= prev + mid for i in range( prev+high-1 - mid, prev+high-1)):
            low = mid+1
        else:
            high = mid
    if all( arr[i] <= prev + low for i in range( prev+high-1 - low, prev+high-1)):
        return prev+low-1
    return prev+low-2

N, K = map( int, input().split() )
P = list( map( int, input().split() ) )

P_idx = [ 0 for _ in range( N+1 ) ]
for i, val in enumerate(P):
    P_idx[ val ] = i # Get the initial index of values

max_right_pos = 0
min_diff = 1 << 30
max_st = [ 0 ]*(N+1)
max_find_func = max_st.__getitem__
FENWICK_SIZE = len(max_st)

def fen_add(idx, val):
    idx += 1
    while idx < FENWICK_SIZE:
        max_st[idx] = max(max_st[idx], val)
        idx += idx & (-idx)

def fen_query_max(left, right):
    result = 0
    while left < right:
        if right & 1:
            right -= 1
            result = max(result, max_find_func(right))
        right >>= 1
        if left & 1:
            result = max(result, max_find_func(left))
            left += 1
        left >>= 1
    return result

for i in range(N-1, -1, -1):
    idx = P_idx[ i+1 ]
    if idx > 0:
        max_left_pos = fen_query_max( idx-K+1, idx )
    else:
        max_left_pos = 0
    fen_add( idx, idx + K )
    max_right_pos = get_max_right_pos( P, idx, max_right_pos )
    min_diff = min( min_diff, max_right_pos - max_left_pos )

print(min_diff)