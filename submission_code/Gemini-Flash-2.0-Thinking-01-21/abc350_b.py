def solve():
    n, q = map(int, input().split())
    treatments = list(map(int, input().split()))
    has_tooth = [True] * n
    for treatment_hole in treatments:
        hole_index = treatment_hole - 1
        has_tooth[hole_index] = not has_tooth[hole_index]
    
    teeth_count = 0
    for tooth_present in has_tooth:
        if tooth_present:
            teeth_count += 1
            
    print(teeth_count)

if __name__ == '__main__':
    solve()