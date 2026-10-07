#!/usr/bin/env python3
"""Finite correctness checks for the restriction-cover reduction.

Uses Python's exact integers and cubic min-plus products. This verifies
identities, complete covers, binary-plan recovery, and the layer-wise DP;
it does not implement or benchmark the external subcubic algorithm.

Run: python3 verify_reduction.py
Only the Python standard library is required. None represents +infinity.
"""
from fractions import Fraction
from itertools import product
from random import Random


def submasks(mask):
    sub = mask
    while True:
        yield sub
        if not sub:
            return
        sub = (sub - 1) & mask


def finite_min(values):
    values = [value for value in values if value is not None]
    return min(values) if values else None


def add(a, b):
    return None if a is None or b is None else a + b


def compatible(restriction, membership):
    # Labels name the forbidden membership: 0=A, 1=B, 2=C.
    return all(x != y for x, y in zip(restriction, membership))


def restriction_masks(word):
    y = sum(1 << i for i, c in enumerate(word) if c == 0)
    z = sum(1 << i for i, c in enumerate(word) if c == 1)
    x = sum(1 << i for i, c in enumerate(word) if c == 2)
    return x, y, z


def ceil_nth_root(value, degree):
    low, high = 0, 1 << ((value.bit_length() + degree - 1) // degree)
    while low + 1 < high:
        middle = (low + high) // 2
        if middle ** degree >= value:
            high = middle
        else:
            low = middle
    return high


def local_cover(r, delta_numerator=1, delta_denominator=500):
    """The manuscript's dyadically rounded weighted-greedy construction."""
    if r == 0:
        return [()]
    a, d = delta_numerator, delta_denominator
    assert 0 < a < d
    rounded = [ceil_nth_root(1 << (d * r - a * k), d)
               for k in range(r + 1)]
    words = list(product(range(3), repeat=r))
    incidence = [
        sum(1 << j for j, target in enumerate(words)
            if compatible(word, target))
        for word in words
    ]
    weights = [sum(rounded[word.count(c)] for c in range(3))
               for word in words]
    full = (1 << len(words)) - 1
    unseen = full
    selected = []
    selected_cost = 0
    while unseen:
        best, best_gain = None, 0
        for i, mask in enumerate(incidence):
            gain = (mask & unseen).bit_count()
            if gain and (best is None
                         or weights[i] * best_gain < weights[best] * gain):
                best, best_gain = i, gain
        assert best is not None
        selected.append(words[best])
        selected_cost += weights[best]
        unseen &= ~incidence[best]
    assert all(any(compatible(word, target) for word in selected)
               for target in words)
    # Exact LP charging check for the rounded integer weights.
    harmonic = sum((Fraction(1, i) for i in range(1, len(words) + 1)),
                   Fraction(0))
    fractional_cost = Fraction(sum(weights), 1 << r)
    assert selected_cost <= harmonic * fractional_cost
    return selected


def product_cover(n, cache, block_size=3):
    sizes = [min(block_size, n - i) for i in range(0, n, block_size)]
    return [
        tuple(label for block in combination for label in block)
        for combination in product(*(cache[size] for size in sizes))
    ]


def restricted_convolution(f, g, n, family):
    result = [None] * (1 << n)
    for word in family:
        x, y, z = restriction_masks(word)
        rows, middle, cols = list(submasks(z)), list(submasks(x)), list(submasks(y))
        f_matrix = [[f[u | t] for t in middle] for u in rows]
        g_matrix = [[g[(x ^ t) | v] for v in cols] for t in middle]
        for i, u in enumerate(rows):
            for j, v in enumerate(cols):
                value = finite_min(add(f_matrix[i][k], g_matrix[k][j])
                                   for k in range(len(middle)))
                output = x | u | v
                result[output] = finite_min((result[output], value))
    return result


def naive_convolution(f, g, n):
    return [finite_min(add(f[a], g[s ^ a]) for a in submasks(s))
            for s in range(1 << n)]


def check_restricted_bijections(max_n=5):
    tested = 0
    for n in range(max_n + 1):
        full = (1 << n) - 1
        for word in product(range(3), repeat=n):
            x, y, z = restriction_masks(word)
            produced = [
                (u | t, (x ^ t) | v)
                for u in submasks(z)
                for t in submasks(x)
                for v in submasks(y)
            ]
            assert len(produced) == len(set(produced)) == 1 << n
            expected = {
                (a, b)
                for a in submasks(full)
                for b in submasks(full ^ a)
                if not (a & y or b & z or (full ^ (a | b)) & x)
            }
            assert set(produced) == expected
            tested += 1
    return tested


def connected(mask, adjacency):
    if not mask:
        return False
    reached = mask & -mask
    while True:
        expanded = reached
        for i, neighbors in enumerate(adjacency):
            if reached >> i & 1:
                expanded |= neighbors & mask
        if expanded == reached:
            return reached == mask
        reached = expanded


def join_dp(costs, n, family, adjacency=None, use_reduction=True):
    dp = [None] * (1 << n)
    for i in range(n):
        dp[1 << i] = 0
    for k in range(2, n + 1):
        conv = restricted_convolution(dp, dp, n, family) if use_reduction else None
        for s in range(1 << n):
            if s.bit_count() != k:
                continue
            if adjacency is not None and not connected(s, adjacency):
                continue
            value = conv[s] if use_reduction else finite_min(
                add(dp[a], dp[s ^ a]) for a in submasks(s) if a and a != s)
            dp[s] = None if value is None else costs[s] + value
    return dp


def recover_tree(s, dp, costs):
    if s & (s - 1) == 0:
        return s.bit_length() - 1
    for a in submasks(s):
        if a and a != s and add(dp[a], dp[s ^ a]) == dp[s] - costs[s]:
            return (recover_tree(a, dp, costs), recover_tree(s ^ a, dp, costs))
    raise AssertionError("No optimal binary split found")


def check_tree(tree, costs, adjacency=None):
    if isinstance(tree, int):
        return 1 << tree, 0
    assert isinstance(tree, tuple) and len(tree) == 2
    left, left_cost = check_tree(tree[0], costs, adjacency)
    right, right_cost = check_tree(tree[1], costs, adjacency)
    assert not left & right
    union = left | right
    if adjacency is not None:
        assert connected(union, adjacency)
        assert any(adjacency[i] & right
                   for i in range(len(adjacency)) if left >> i & 1)
    return union, left_cost + right_cost + costs[union]


def main():
    rng = Random(20261006)
    bijections = check_restricted_bijections()
    print(f"PASS: {bijections} exhaustive restricted-product bijections (n <= 5).")
    cache = {r: local_cover(r) for r in range(1, 6)}
    print("PASS: complete local covers and exact greedy/LP bounds:",
          {r: len(family) for r, family in cache.items()})

    convolution_cases = 0
    for n in range(8):
        family = product_cover(n, cache)
        assert all(any(compatible(word, target) for word in family)
                   for target in product(range(3), repeat=n))
        for trial in range(20):
            bits = 4096 if trial == 0 else 12
            def value():
                if rng.random() < .15:
                    return None
                return rng.getrandbits(bits) * rng.choice((-1, 1))
            f, g = [[value() for _ in range(1 << n)] for _ in range(2)]
            assert restricted_convolution(f, g, n, family) == naive_convolution(f, g, n)
            convolution_cases += 1
    print(f"PASS: {convolution_cases} exact convolution comparisons, including 4096-bit values.")

    join_cases = 0
    recovered = 0
    for n in range(2, 8):
        family = product_cover(n, cache)
        for trial in range(12):
            adjacency = [0] * n
            for i in range(n):
                for j in range(i):
                    if rng.random() < .45:
                        adjacency[i] |= 1 << j
                        adjacency[j] |= 1 << i
            costs = [rng.getrandbits(256 if trial == 0 else 12)
                     for _ in range(1 << n)]
            for graph in (None, adjacency):
                fast = join_dp(costs, n, family, graph, True)
                slow = join_dp(costs, n, family, graph, False)
                assert fast == slow
                if fast[-1] is not None:
                    tree = recover_tree((1 << n) - 1, fast, costs)
                    mask, tree_cost = check_tree(tree, costs, graph)
                    assert mask == (1 << n) - 1 and tree_cost == fast[-1]
                    recovered += 1
                join_cases += 1
    print(f"PASS: {join_cases} join-DP comparisons; {recovered} recovered binary plans verified.")
    print("All checks passed. No subcubic running time is claimed for this verification script.")


if __name__ == "__main__":
    main()
