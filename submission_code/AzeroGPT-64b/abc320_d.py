"""kattis - buildingbridges

Problem taken from Kattis:
https://open.kattis.com/problems/buildingbridges

Solution by Rakesh Udelwal https://github.com/fr4ncklin
"""
import sys
from collections import defaultdict
from math import gcd

N, M = map(int, sys.stdin.readline().split())
points  = [None] * (N + 1)
vectors = defaultdict(lambda: defaultdict(list))

xi = []
for _ in range(M) :
    ai, bi, xi0, yi0 = map(int, sys.stdin.readline().split())
    vectors[ai][bi].append((xi0, yi0))
    xi.append(xi0)
    xi.append(yi0)
    
if any([point is None for point in points[1:]] and [x != 0 for x in xi]) :

    scale  = abs(max(xi)) if xi else 1
    x, y   = 0, 0
    points = [(x, y)]

    for i in range(1, N + 1) :

        if points[i - 1] is not None :

            eai = points[i - 1]
            exi = defaultdict(lambda: defaultdict(list))
            
            for ai, _ in vectors.items() :
                for bi, vectors0 in _.items() :
                    if ai == i :

                        for (xi0, yi0) in vectors0 :
                            exi[bi][ai].append((xi0, yi0))

                    elif bi == i :

                        for (xi0, yi0) in vectors0 :
                            exi[ai][bi].append((yi0, -xi0))

            ex  = 1
            ey  = 0
            e0a = (float('inf'), float('inf'))
            e0s = None
            e0t = None
            
            for aj, _ in exi.items() :

                if points[aj] is None : continue
                
                eaj = points[aj]
                e0jt = eaj[:2]
                e0j  = (eaj[1], -eaj[0])

                for ay, bx in _.items() : 

                    if points[ay] is None : continue
                    
                    eya = points[ay]

                    for (eyxi, eyyi) in bx :      

                        e0x = eya[0] + eyxi - eaj[0]
                        e0y = eya[1] + eyyi - eaj[1]

                        e0i = (e0x, e0y)
                        e0pull = eyxi * e0j[1] - eyyi * (-e0j[0]) 

                        if e0pull != 0 :

                            ex = e0j[1] / e0pull
                            ey = -e0j[0] / e0pull

                            e0s = e0x - eyyi * ex + eyxi * ey
                            e0t = e0y + eyxi * ex + eyyi * ey
                            
                            if abs(e0s) + abs(e0t) < abs(e0a[0]) + abs(e0a[1]) :

                                e0a = (e0s, e0t)
                            
            is_zero = all([not ex and not ey, abs(e0a[0]) >= scale and abs(e0a[1]) >= scale])
            if (ex, ey) == (1, 0) and e0a == (0, 0) :

                is_zero = True

            if is_zero :

                print('undecidable')
            else :

                ex = int(gcd(ex, ey))
                ey = int(gcd(ex, ey))

                e0a = (e0a[0] / gcd(e0a[0], e0a[1]), e0a[1] / gcd(e0a[0], e0a[1]))

                points[i] = (int(round(e0a[0] * ex))
                    , int(round(e0a[1] * ey)))

        if points[i] is None :

            print('undecidable')
        else :

            print('{0} {1}'.format(*points[i]))

else :

    print('undecidable' * N)