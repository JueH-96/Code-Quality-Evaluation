from typing import List
import itertools

class SolBrute:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        n=len(enemyEnergies)
        best=0
        # Represent state by tuple (energy, points, maskMarked)
        from collections import deque
        start=(currentEnergy,0,0)
        seen=set([start])
        q=deque([start])
        while q:
            e,p,mark=q.popleft()
            best=max(best,p)
            for i,val in enumerate(enemyEnergies):
                mask_bit=1<<i
                if mark & mask_bit: # already marked
                    continue
                # attack only
                if e>=val:
                    new=(e-val,p+1,mark)  # still unmarked
                    if new not in seen:
                        seen.add(new)
                        q.append(new)
                # mark
                if p>=1:
                    new=(e+val,p,mark|mask_bit)
                    if new not in seen:
                        seen.add(new)
                        q.append(new)
        return best

def maxPoints_greedy(arr, energy):
    arr=sorted(arr)
    if energy < arr[0]:
        return 0
    core=arr[0]
    energy-=core
    points=1
    i=1
    j=len(arr)-1
    while i<=j:
        if energy>=arr[i]:
            points+=1
            i+=1
        else:
            energy+=arr[j]
            j-=1
    points+=energy//core
    return points

def test():
    import random
    for n in range(1,9):
        for _ in range(500):
            arr=[random.randint(1,5) for _ in range(n)]
            energy=random.randint(0,10)
            b=SolBrute().maximumPoints(arr, energy)
            g=maxPoints_greedy(arr, energy)
            if b!=g:
                print("Mismatch",arr,energy,b,g)
                return
    print("all good")
test()