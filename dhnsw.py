import pprint
from heapq import heapify, heappop, heappush, heapreplace, nlargest, nsmallest
from math import log2
from operator import itemgetter
from random import random
import numpy as np
import time
import pandas as pd

class HNSW(object):
    def l2_distance(self, a, b):
        return np.linalg.norm(a - b)

    def cosine_distance(self, a, b):
        try:
            return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        except ValueError:
            print(a)
            print(b)

    def _distance(self, x, y):
        return self.distance_func(x, [y])[0]

    def vectorized_distance_(self, x, ys):
        start_time = time.time()
        distances = [self.distance_func(x, y) for y in ys]
        end_time = time.time()
        self.time_measurements["distance_calculation"] += end_time - start_time
        return distances

    def __init__(self, distance_type, m=5, ef=200, m0=None, heuristic=True, vectorized=False):
        self.data = []
        if distance_type == "l2":
            distance_func = self.l2_distance
        elif distance_type == "cosine":
            distance_func = self.cosine_distance
        else:
            raise TypeError('Please check your distance type!')

        self.distance_func = distance_func

        if vectorized:
            self.distance = self._distance
            self.vectorized_distance = distance_func
        else:
            self.distance = distance_func
            self.vectorized_distance = self.vectorized_distance_

        self._m = m
        self._ef = ef
        self._m0 = 2 * m if m0 is None else m0
        self._level_mult = 1 / log2(m)
        self._graphs = []
        self._enter_point = None

        self._select = (
            self._select_heuristic if heuristic else self._select_naive)

        self.time_measurements = {
            "add": 0,
            "distance_calculation": 0,
            "other_operations": 0,
            "_search_graph_ef1": 0,
            "_search_graph": 0,
            "search": 0  # 'search' 키 추가
        }

    def add(self, elem, ef=None):
        start_time = time.time()

        if ef is None:
            ef = self._ef

        distance = self.distance
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        m = self._m

        level = int(-log2(random()) * self._level_mult) + 1

        idx = len(data)
        data.append(elem)

        if point is not None:
            dist_start_time = time.time()
            dist = distance(elem, data[point])
            dist_end_time = time.time()
            self.time_measurements["distance_calculation"] += dist_end_time - dist_start_time

            for layer in reversed(graphs[level:]):
                point, dist = self._search_graph_ef1(elem, point, dist, layer)

            ep = [(-dist, point)]
            layer0 = graphs[0]
            for layer in reversed(graphs[:level]):
                level_m = m if layer is not layer0 else self._m0
                ep = self._search_graph(elem, ep, layer, ef)
                layer[idx] = layer_idx = {}
                self._select(layer_idx, ep, level_m, layer, heap=True)
                for j, dist in layer_idx.items():
                    self._select(layer[j], (idx, dist), level_m, layer)
        for i in range(len(graphs), level):
            graphs.append({idx: {}})
            self._enter_point = idx

        end_time = time.time()
        self.time_measurements["add"] += end_time - start_time
        self.time_measurements["other_operations"] += end_time - start_time - self.time_measurements["distance_calculation"]

    def balanced_add(self, elem, ef=None):
        if ef is None:
            ef = self._ef

        distance = self.distance
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        m = self._m
        m0 = self._m0

        idx = len(data)
        data.append(elem)

        if point is not None:
            dist_start_time = time.time()
            dist = distance(elem, data[point])
            dist_end_time = time.time()
            self.time_measurements["distance_calculation"] += dist_end_time - dist_start_time

            pd = [(point, dist)]
            for layer in reversed(graphs[1:]):
                point, dist = self._search_graph_ef1(elem, point, dist, layer)
                pd.append((point, dist))
            for level, layer in enumerate(graphs):
                level_m = m0 if level == 0 else m
                candidates = self._search_graph(
                    elem, [(-dist, point)], layer, ef)
                layer[idx] = layer_idx = {}
                self._select(layer_idx, candidates, level_m, layer, heap=True)
                for j, dist in layer_idx.items():
                    self._select(layer[j], [idx, dist], level_m, layer)
                    assert len(layer[j]) <= level_m
                if len(layer_idx) < level_m:
                    return
                if level < len(graphs) - 1:
                    if any(p in graphs[level + 1] for p in layer_idx):
                        return
                point, dist = pd.pop()
        graphs.append({idx: {}})
        self._enter_point = idx

    def search(self, q, k=None, ef=None):
        start_time = time.time()
    
        distance = self.distance
        graphs = self._graphs
        point = self._enter_point
    
        if ef is None:
            ef = self._ef
    
        if point is None:
            raise ValueError("Empty graph")
    
        dist_start_time = time.time()
        dist = distance(q, self.data[point])
        dist_end_time = time.time()
        self.time_measurements["distance_calculation"] += dist_end_time - dist_start_time
    
        for layer in reversed(graphs[1:]):
            point, dist = self._search_graph_ef1(q, point, dist, layer)
        ep = self._search_graph(q, [(-dist, point)], graphs[0], max(ef, k))
    
        if k is not None:
            ep = nlargest(k, ep)
        else:
            ep.sort(reverse=True)
    
        end_time = time.time()
        self.time_measurements["search"] += end_time - start_time
    
        return [(idx, -md) for md, idx in ep]

    def _search_graph_ef1(self, q, entry, dist, layer):
        start_time = time.time()

        vectorized_distance = self.vectorized_distance
        data = self.data

        best = entry
        best_dist = dist
        candidates = [(dist, entry)]
        visited = set([entry])

        while candidates:
            dist, c = heappop(candidates)
            if dist > best_dist:
                break
            edges = [e for e in layer[c] if e not in visited]
            visited.update(edges)
            dists = vectorized_distance(q, [data[e] for e in edges])
            for e, dist in zip(edges, dists):
                if dist < best_dist:
                    best = e
                    best_dist = dist
                    heappush(candidates, (dist, e))

        end_time = time.time()
        self.time_measurements["_search_graph_ef1"] += end_time - start_time

        return best, best_dist

    def _search_graph(self, q, ep, layer, ef):
        start_time = time.time()
    
        vectorized_distance = self.vectorized_distance
        data = self.data
    
        candidates = [(-mdist, p) for mdist, p in ep]
        heapify(candidates)
        visited = set(p for _, p in ep)
    
        while candidates:
            dist, c = heappop(candidates)
            mref = ep[0][0]
            if dist > -mref:
                break
            edges = [e for e in layer[c] if e not in visited]
            visited.update(edges)
            dists = vectorized_distance(q, [data[e] for e in edges])
            for e, dist in zip(edges, dists):
                mdist = -dist
                if len(ep) < ef:
                    heappush(candidates, (dist, e))
                    heappush(ep, (mdist, e))
                    mref = ep[0][0]
                elif mdist > mref:
                    heappush(candidates, (dist, e))
                    heapreplace(ep, (mdist, e))
                    mref = ep[0][0]
    
        end_time = time.time()
        self.time_measurements["_search_graph"] += end_time - start_time
    
        return ep

    def _select_naive(self, d, to_insert, m, layer, heap=False):
        if not heap:
            idx, dist = to_insert
            assert idx not in d
            if len(d) < m:
                d[idx] = dist
            else:
                max_idx, max_dist = max(d.items(), key=itemgetter(1))
                if dist < max_dist:
                    del d[max_idx]
                    d[idx] = dist
            return

        assert not any(idx in d for _, idx in to_insert)
        to_insert = nlargest(m, to_insert)
        unchecked = m - len(d)
        assert 0 <= unchecked <= m
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)
        if to_check > 0:
            checked_del = nlargest(to_check, d.items(), key=itemgetter(1))
        else:
            checked_del = []
        for md, idx in to_insert:
            d[idx] = -md
        zipped = zip(checked_ins, checked_del)
        for (md_new, idx_new), (idx_old, d_old) in zipped:
            if d_old <= -md_new:
                break
            del d[idx_old]
            d[idx_new] = -md_new
            assert len(d) == m

    def _select_heuristic(self, d, to_insert, m, g, heap=False):
        nb_dicts = [g[idx] for idx in d]
    
        def prioritize(idx, dist):
            return any(nd.get(idx, float('inf')) < dist for nd in nb_dicts), dist, idx
    
        if not heap:
            idx, dist = to_insert
            to_insert = [prioritize(idx, dist)]
        else:
            to_insert = nsmallest(m, (prioritize(idx, -mdist)
                                      for mdist, idx in to_insert))
    
        assert len(to_insert) > 0
        assert not any(idx in d for _, _, idx in to_insert)
    
        unchecked = max(0, min(m - len(d), len(to_insert)))
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)
        if to_check > 0:
            checked_del = nlargest(to_check, (prioritize(idx, dist)
                                              for idx, dist in d.items()))
        else:
            checked_del = []
        for _, dist, idx in to_insert:
            d[idx] = dist
        zipped = zip(checked_ins, checked_del)
        for (p_new, d_new, idx_new), (p_old, d_old, idx_old) in zipped:
            if (p_old, d_old) <= (p_new, d_new):
                break
            del d[idx_old]
            d[idx_new] = d_new
        
        if len(d) > m:
            d = dict(sorted(d.items(), key=lambda item: item[1])[:m])

    def __getitem__(self, idx):
        for g in self._graphs:
            try:
                yield from g[idx].items()
            except KeyError:
                return

    def get_average_neighbors(self):
        """
        그래프의 각 노드가 연결된 이웃 수를 계산하고 평균 이웃 수를 반환합니다.
        """
        total_neighbors = 0  # 전체 이웃 수의 합
        total_nodes = 0  # 전체 노드 수
    
        for level, graph in enumerate(self._graphs):
            for node_id, neighbors in graph.items():
                total_neighbors += len(neighbors)  # 이웃 수 누적
                total_nodes += 1  # 노드 수 누적
    
        if total_nodes == 0:
            return 0  # 노드가 없으면 0 반환
    
        average_neighbors = total_neighbors / total_nodes  # 평균 이웃 수 계산
        return average_neighbors

    def get_average_neighbors_per_level(self):
        """
        각 레벨별 평균 이웃 수를 계산하고 출력합니다.
        """
        for level, graph in enumerate(self._graphs):
            total_neighbors = sum(len(neighbors) for neighbors in graph.values())
            total_nodes = len(graph)
            avg_neighbors = total_neighbors / total_nodes if total_nodes > 0 else 0
            print(f"Level {level}: Average neighbors per node: {avg_neighbors:.2f}")
    

class DynamicHNSW(HNSW):
    def __init__(self, distance_type, densities, m_start=5, m_end=32, ef_start=10, ef_end=200, m0=None, heuristic=True, vectorized=False, invert_density=False):
        super().__init__(distance_type, m=m_start, ef=ef_start, m0=m0, heuristic=heuristic, vectorized=vectorized)
        self.m_start = m_start
        self.m_end = m_end
        self.ef_start = ef_start
        self.ef_end = ef_end
        self.densities = densities
        self.invert_density = invert_density
        self.min_density = min(densities)
        self.max_density = max(densities)
        # self.ef_history = []  # ef 값의 변화를 추적하기 위한 리스트
        # self.m_history = []   # M 값의 변화를 추적하기 위한 리스트

    def _get_dynamic_ef(self, index):
        density = self.densities[index]
        if self.invert_density:
            dynamic_ef = self.ef_end - (density - self.min_density) / (self.max_density - self.min_density) * (self.ef_end - self.ef_start)
        else:
            dynamic_ef = self.ef_start + (density - self.min_density) / (self.max_density - self.min_density) * (self.ef_end - self.ef_start)
        
        dynamic_ef = int(min(self.ef_end, max(self.ef_start, dynamic_ef)))
        # self.ef_history.append(dynamic_ef)  # 주석 처리
        return dynamic_ef

    def _get_dynamic_m(self, index):
        density = self.densities[index]
        if self.invert_density:
            dynamic_m = self.m_end - (density - self.min_density) / (self.max_density - self.min_density) * (self.m_end - self.m_start)
        else:
            dynamic_m = self.m_start + (density - self.min_density) / (self.max_density - self.min_density) * (self.m_end - self.m_start)
        
        dynamic_m = int(min(self.m_end, max(self.m_start, dynamic_m)))
        # self.m_history.append(dynamic_m)  # 주석 처리
        return dynamic_m

    def add(self, elem, ef=None):
        dynamic_m = self._get_dynamic_m(len(self.data))
        self._m = dynamic_m  # 동적 M 값 설정
        if ef is None:
            ef = self._get_dynamic_ef(len(self.data))
        super().add(elem, ef)

    def balanced_add(self, elem, ef=None):
        dynamic_m = self._get_dynamic_m(len(self.data))
        self._m = dynamic_m  # 동적 M 값 설정
        if ef is None:
            ef = self._get_dynamic_ef(len(self.data))
        super().balanced_add(elem, ef)

    def search(self, q, k=None, ef=None):
        start_time = time.time()

        distance = self.distance
        graphs = self._graphs
        point = self._enter_point

        if ef is None:
            ef = self._ef

        if point is None:
            raise ValueError("Empty graph")

        dist_start_time = time.time()
        dist = distance(q, self.data[point])
        dist_end_time = time.time()
        self.time_measurements["distance_calculation"] += dist_end_time - dist_start_time

        for layer in reversed(graphs[1:]):
            point, dist = self._search_graph_ef1(q, point, dist, layer)
        ep = self._search_graph(q, [(-dist, point)], graphs[0], max(ef, k))

        if k is not None:
            ep = nlargest(k, ep)
        else:
            ep.sort(reverse=True)

        end_time = time.time()
        self.time_measurements["search"] += end_time - start_time

        return [(idx, -md) for md, idx in ep]
