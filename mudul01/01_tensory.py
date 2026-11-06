# %%

# Czym jest tensor w PyTorch?

from tracemalloc import start

import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]])

print(x)
print("shape:", x.shape)
print("ndim:", x.ndim)
print("dtype:", x.dtype)
print("device:", x.device)


# %%

# Tworzenie tensorów

print(torch.zeros((2, 3)))  # tensor wypełniony zerami
print(torch.ones((2, 3)))  # tensor wypełniony jedynkami
print(torch.rand((2, 3)))  # tensor wypełniony losowymi liczbami z przedziału [0, 1)
print(torch.randn((2, 3)))  # tensor wypełniony losowymi liczbami z rozkładu normalnego
print(torch.arange(0, 10, 2))  # tensor z liczbami od 0 do 10 z krokiem 2
print(torch.linspace(0, 1, 5))  # tensor z 5 liczbami rozłożonymi między 0 a 1

torch.manual_seed(42)  # ustawienie ziarna generatora liczb dla powtarzalności wyników
print(torch.rand(2))  # tensor wypełniony losowymi liczbami z przedziału [0, 1)


# %%
print(x.float())  # konwersja typu danych na float
print(x.to(torch.float16))  # konwersja typu danych na float16
print(x)


# %%

# Kształt, indeksowanie, przekształcanie tensorów

a = torch.arange(12)
b = a.reshape(3, 4)  # zmiana kształtu tensora na 3x4
print(b)
print(b[0])  # indeksowanie pierwszego wiersza
print(b[:, 1])  # indeksowanie drugiej kolumny
print(b[1:, :2])
print(b[b > 5])  # indeksowanie elementów większych niż 5
print(b >= 5)  # maska logiczna
print(b.T)  # transpozycja tensora


# %%

# Operacje i broadcasting

u = torch.tensor([1.0, 2.0, 3.0])
v = torch.tensor([10.0, 20.0, 30.0])
print(u + v, u * v)  # działa element po elemencie
print(u @ v)  # iloczyn skalarny

m = torch.randn(3, 4)
print(m)
# sumowanie po wszystkich elementach, po wierszach i po kolumnach
print(m.sum(), m.sum(dim=0), m.sum(dim=1))
print(m.mean(), m.max(), m.min())


# %%

# Broadcasting

col = torch.tensor([[1], [2], [3]])
row = torch.tensor([[10, 20, 30, 40]])
print(col + row)  # broadcasting


# %%

# NumPy i GPU

import numpy as np

arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)  # konwersja z NumPy do PyTorch
x = torch.tensor(arr)  # tworzenie tensora z tablicy NumPy
print(t)
print(x)
arr[0] = 99.0  # zmiana wartości w tablicy NumPy
print(t)  # tensor t również się zmienił, ponieważ jest powiązany z tablicą NumPy
print(x)  # tensor x nie zmienił się, ponieważ został utworzony niezależnie


device = "cuda" if torch.cuda.is_available() else "cpu"  # sprawdzenie dostępności GPU
# tworzenie tensora na GPU lub CPU w zależności od dostępności
g = torch.rand(3, 3).to(device)
print(g)
print(g.device)  # wyświetlenie urządzenia, na którym znajduje się tensor g

# %%

# Porównanie wydajności CPU i GPU

import time


def bench(device, n=4096, reps=10):
    x = torch.rand(n, n, device=device)
    y = torch.rand(n, n, device=device)
    # synchronizacja przed rozpoczęciem pomiaru czasu
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(reps):
        z = x @ y  # mnożenie macierzy
    # synchronizacja po zakończeniu pomiaru czasu
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.time()
    return end - start


print("CPU time:", bench("cpu"), "s")
print("GPU time:", bench("cuda"), "s")  # pomiar czasu na GPU


# %%

# Ćwiczenia

import time, torch

# 1. Utwórz tensor (4, 5) z liczb od 0 do 19 i wypisz:
# ostatni wiersz, trzecią kolumnę oraz element z drugiego wiersza i czwartej kolumny.
print("Zadanie 1")
l = torch.arange(20).reshape(4, 5)
print(l)
print(l[-1])  # ostatni wiersz
print(l[:, 2])  # trzecia kolumna
print(l[1, 3])  # element z drugiego wiersza i czwartej kolumny


# 2. Zamień tensor z zadania 1 na float32 i policz średnią każdej kolumny.
# Dlaczego mean nie zadziała na int64?
print("Zadanie 2")
lf = l.to(torch.float32)
print(lf.mean(dim=0))


# 3. Bez uruchamiania przewidź kształt wyniku:
# torch.ones(5, 1, 3) + torch.ones(4, 1). Potem sprawdź.
print("Zadanie 3")
print(torch.ones(5, 1, 3) + torch.ones(4, 1))


# 4. Wygeneruj torch.randn(1000, 1000),policz, ile elementów jest większych od 1
# (wskazówka: maska i .sum()).
# Jaki ułamek całości to jest i czy zgadza się z oczekiwaniem dla rozkładu normalnego (ok. 16%)?
print("Zadanie 4")
t = torch.randn(1000, 1000)
mask = t > 1
count = mask.sum().item()
fraction = count / t.numel()
print(f"Count of elements > 1: {count}, Fraction: {fraction:.4%}")


# 5. Zmodyfikuj bench, by porównać float32 z float16 na GPU. Co zauważasz?

print("Zadanie 5")


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def bench(device, dtype, n=4096, reps=50, warmup=3):
    x = torch.rand(n, n, device=device, dtype=dtype)
    y = torch.rand(n, n, device=device, dtype=dtype)

    for _ in range(warmup):  # rozgrzewka przed pomiarem czasu
        x @ y

    # synchronizacja przed rozpoczęciem pomiaru czasu
    sync(device)
    start = time.perf_counter()

    for _ in range(reps):
        z = x @ y  # mnożenie macierzy

    # synchronizacja po zakończeniu pomiaru czasu
    sync(device)
    end = time.perf_counter()
    return (end - start) / reps


print("CPU time (float32): ", bench("cpu", torch.float32, n=1024), "s")
print("CPU time (float16): ", bench("cpu", torch.float16, n=1024), "s")
print("CPU time (bfloat16): ", bench("cpu", torch.bfloat16, n=1024), "s")
print("GPU time (float32): ", bench("cuda", torch.float32), "s")  # pomiar czasu na GPU
print("GPU time (float16): ", bench("cuda", torch.float16), "s")  # pomiar czasu na GPU
print("GPU time (bfloat16): ", bench("cuda", torch.bfloat16), "s")  # pomiar czasu na GPU

# %%
