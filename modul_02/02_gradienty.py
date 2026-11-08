# %%

# 1. Pierwszy gradient

import torch

x = torch.tensor(3.0, requires_grad=True)  # śledź operacje na x
y = x**2 + 2 * x  # funkcja kwadratowa y
y.backward()  # obliczenie gradientu (pochodnej) y względem x
# wyświetlenie gradientu (pochodnej) y względem x, który wynosi 2*x + 2 = 8.0 dla x=3
print(x.grad)

# %%

# 2. Gradienty się sumują

x = torch.tensor(3.0, requires_grad=True)
for _ in range(3):
    y = x**2 + 2 * x
    y.backward()  # obliczenie gradientu (pochodnej) y względem x
    print(x.grad)  # wyświetlenie gradientu (pochodnej) y względem x

x.grad.zero_()  # zerowanie gradientu przed kolejną iteracją

# %%

# 3. Wyłączanie śledzenia gradientów

x = torch.tensor(3.0, requires_grad=True)
with torch.no_grad():  # wyłączenie śledzenia operacji na x
    z = x * 2
print(z)  # wyświetlenie wyniku operacji na x bez śledzenia gradientu
print(z.requires_grad)
w = x.detach()  # odłączenie tensora z grafu obliczeń
print(w.requires_grad)  # wyświetlenie informacji, czy tensor w wymaga gradientu

# %%

# 4. Ręczna regresja liniowa
# Dane y=3x+2+ szum, gdzie x to dane wejściowe, a y to dane wyjściowe. Celem jest znalezienie współczynników a i b w równaniu y=ax+b.

torch.manual_seed(0)  # ustawienie ziarna losowości dla powtarzalności wyników
# Generowanie danych
X = torch.rand(100, 1) * 10  # 100 losowych
y = 3 * X + 2 + torch.randn(100, 1)  # dodanie szumu do danych wyjściowych

w = torch.zeros(1, requires_grad=True)  # inicjalizacja wagi (a) ze śledzeniem gradientu
b = torch.zeros(1, requires_grad=True)  # inicjalizacja biasu (b) ze śledzeniem gradientu
lr = 0.001  # współczynnik uczenia
lr = 0.03  # współczynnik uczenia
# lr = 0.05  # współczynnik uczenia
for epoch in range(2000):  # pętla uczenia przez 2000 epok
    pred = X * w + b  # obliczenie predykcji na podstawie aktualnych wag i biasu
    # obliczenie straty jako średniej różnicy między rzeczywistymi a przewidywanymi wartościami
    loss = ((y - pred) ** 2).mean()  # błąd średniokwadratowy (MSE)
    loss.backward()  # obliczenie gradientu straty względem wag i biasu
    with torch.no_grad():  # wyłączenie śledzenia gradientów podczas aktualizacji wag
        w -= lr * w.grad  # aktualizacja wagi (a) na podstawie gradientu
        b -= lr * b.grad  # aktualizacja biasu (b) na podstawie gradientu
    # w.grad.zero_()  # zerowanie gradientu wagi przed kolejną iteracją
    # b.grad.zero_()  # zerowanie gradientu biasu przed kolejną iteracją
    if epoch % 100 == 0:  # co 100 epok wyświetl informację o postępie uczenia
        print(f"Epoch {epoch}: loss={loss.item()}, w={w.item()}, b={b.item()}")

print(f"\nLoss: {loss.item()}")
print(f"Final weights: w={w.item()}, b={b.item()}")  # wyświetlenie finalnych wag

# %%

# Zadanie 1.
# Policz ręcznie pochodną f(x)=x³ w punkcie x=2 (wynik 12) i potwierdź przez autograd

x = torch.tensor(2.0, requires_grad=True)
y = x**3
y.backward()
# wyświetlenie gradientu (pochodnej) y względem x, który wynosi 3*x^2 = 12.0 dla x=2
print(x.grad)

# Zadanie 2.
# Dla wektora: x = torch.tensor([1., 2., 3.], requires_grad=True), y = (x**2).sum()
# Zgadnij x.grad przed uruchomieniem

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x**2).sum()  # suma kwadratów elementów tensora x
y.backward()  # obliczenie gradientu (pochodnej) y względem x
print(x.grad)  # wyświetlenie gradientu (pochodnej) y względem x, który wynosi [2, 4, 6]


# %%

# Oszacowanie wartości

A = torch.cat([X, torch.ones_like(X)], dim=1)  # kolumny: x oraz 1
H = 2 * A.T @ A / len(X)
eig = torch.linalg.eigvalsh(H)  # rosnąco: [λ_min, λ_max]
print(H)
print(
    eig,
    "granica lr:",
    (2 / eig.max()).item(),
    "uwarunkowanie:",
    (eig.max() / eig.min()).item(),
)
# %%
