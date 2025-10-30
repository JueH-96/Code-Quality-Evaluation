def solve():
    n, q = map(int, input().split())
    treatments = list(map(int, input().split()))
    
    teeth_holes = set()
    for i in range(1, n + 1):
        teeth_holes.add(i)
        
    for treatment_hole in treatments:
        if treatment_hole in teeth_holes:
            teeth_holes.remove(treatment_hole)
        else:
            teeth_holes.add(treatment_hole)
            
    print(len(teeth_holes))

if __name__ == '__main__':
    solve()