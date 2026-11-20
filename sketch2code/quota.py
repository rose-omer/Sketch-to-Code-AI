"""Herkese açık demo için günlük üretim sınırı.

Yalnız sahibin anahtarı (st.secrets ya da ortam değişkeni) kullanılırken uygulanır; kullanıcı kendi
anahtarını girdiyse sınır yoktur. Sayaç süreç belleğindedir: yeniden başlatmada sıfırlanır.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from datetime import date

LIMIT_ENV = "DAILY_GENERATION_LIMIT"


def limit_from_env(env: dict[str, str] | None = None) -> int:
    """Ortam değişkeninden sınır; tanımsız, geçersiz ya da 0 ise 0 (sınırsız)."""
    raw = (env if env is not None else os.environ).get(LIMIT_ENV, "").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


class DailyQuota:
    """Ziyaretçi başına günlük sayaç. İş parçacığı güvenli (Streamlit oturumları aynı süreçte koşar)."""

    def __init__(self, limit: int, today: Callable[[], date] = date.today) -> None:
        self.limit = limit
        self._today = today
        self._lock = threading.Lock()
        self._day = today()
        self._used: dict[str, int] = {}

    def _roll(self) -> None:
        """Gün değiştiyse bütün sayaçları sıfırlar (eski günler bellekte birikmez)."""
        today = self._today()
        if today != self._day:
            self._day = today
            self._used.clear()

    def remaining(self, visitor: str) -> int:
        with self._lock:
            self._roll()
            return max(0, self.limit - self._used.get(visitor, 0))

    def consume(self, visitor: str) -> bool:
        """Bir hak düşer; hak kalmadıysa False döner ve sayaç değişmez."""
        with self._lock:
            self._roll()
            count = self._used.get(visitor, 0)
            if count >= self.limit:
                return False
            self._used[visitor] = count + 1
            return True
