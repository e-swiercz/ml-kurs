# Moduł 2: autograd i spadek gradientu

- `requires_grad=True` włącza śledzenie, `loss.backward()` liczy gradienty.
- Gradienty się **sumują**: po każdym kroku `grad.zero_()` (inaczej aktualizacja używa sumy starych gradientów).
- Aktualizację parametrów robimy w `torch.no_grad()`, żeby krok nie trafił do grafu.
- Zbieżność spadku gradientu dla funkcji kwadratowej: `lr < 2 / λ_max` (największa krzywizna hesjanu).
- Tempo wyznacza najmniejsza krzywizna; iloraz `λ_max/λ_min` to uwarunkowanie.
- Normalizacja danych (średnia 0, odchylenie 1) poprawia uwarunkowanie i przyspiesza uczenie.
- Wyniki: lr=0.001 za wolno, lr=0.03 zbieżne (w≈2.93, b≈2.33), lr=0.05 rozbieżne (nan).
