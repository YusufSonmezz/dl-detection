"""
Run Tracker — dl-detection experiment history
Mevcut run'ları kayıt altına alır, notlama ve sorgulama sağlar.

Kullanım:
    python train/run_tracker.py migrate              # Bir kere: mevcut run'ları kaydet
    python train/run_tracker.py list                 # Tüm run'ları listele
    python train/run_tracker.py list --mode fuzzy    # Filtreleme
    python train/run_tracker.py last                 # Son run
    python train/run_tracker.py last 3               # Son 3 run
    python train/run_tracker.py show <name>          # Detay
    python train/run_tracker.py note <name> "<text>" # Not ekle
    python train/run_tracker.py parent <name> <parent_name>  # Parent ata
"""

import json, os, sys, csv, argparse, re
from datetime import datetime
from pathlib import Path

# ============================================================================
#  PATHS  (working dir = proje root)
# ============================================================================
RUNS_DIR     = os.path.join("artifacts", "runs")
HISTORY_FILE = os.path.join(RUNS_DIR, "run_history.json")

# ============================================================================
#  RunScanner  —  mevcut run klasörlerini okur, metadata çıkarır
# ============================================================================
class RunScanner:

    @staticmethod
    def scan_all():
        """artifacts/runs/ altındaki tüm run klasörlerini bul, isim listesi dön."""
        if not os.path.isdir(RUNS_DIR):
            return []
        return sorted([
            d for d in os.listdir(RUNS_DIR)
            if os.path.isdir(os.path.join(RUNS_DIR, d))
        ])

    @staticmethod
    def _json(path):
        """JSON dosyayu oku, hata varsa None dön."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def read_params(name):
        return RunScanner._json(os.path.join(RUNS_DIR, name, "params.json"))

    @staticmethod
    def read_summary(name):
        return RunScanner._json(os.path.join(RUNS_DIR, name, "run_summary.json"))

    @staticmethod
    def recover_map50(name):
        """
        epochs.csv'den best_map50 recovery.
        Header'dan kolonu isimle bul — format versioning'e karşı sağlam.
        """
        csv_path = os.path.join(RUNS_DIR, name, "epochs.csv")
        if not os.path.isfile(csv_path):
            return None
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                best = None
                for row in reader:
                    # best_map50 kolonu varsa ve dolu ise onu tercih et
                    if "best_map50" in row and row["best_map50"]:
                        val = float(row["best_map50"])
                        if val > 0:
                            best = val  # Son satırdaki best_map50 en yüksek olur
                    # Yoksa map50 max'ı takip et
                    elif "map50" in row and row["map50"]:
                        val = float(row["map50"])
                        if best is None or val > best:
                            best = val
                return best
        except Exception:
            return None

    @staticmethod
    def get_timestamps(name):
        """
        started_at: params.json mtime
        finished_at: run_summary.json mtime (yok ise None)
        """
        started = finished = None
        p = os.path.join(RUNS_DIR, name, "params.json")
        if os.path.isfile(p):
            started = datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds")
        s = os.path.join(RUNS_DIR, name, "run_summary.json")
        if os.path.isfile(s):
            finished = datetime.fromtimestamp(os.path.getmtime(s)).isoformat(timespec="seconds")
        return started, finished

    @staticmethod
    def detect_status(name):
        """
        run_summary.json var → complete
        yok, epochs.csv var  → crashed
        sadece params.json   → crashed
        """
        run_path = os.path.join(RUNS_DIR, name)
        if os.path.isfile(os.path.join(run_path, "run_summary.json")):
            return "complete"
        if os.path.isfile(os.path.join(run_path, "epochs.csv")):
            return "crashed"
        if os.path.isfile(os.path.join(run_path, "params.json")):
            return "crashed"
        return "unknown"

    @staticmethod
    def detect_parents(run_names):
        """
        Otomatik parent detection.
        Pattern: X_v{N} → parent = X_v{N-1} (eğer var ise)
        """
        name_set = set(run_names)
        parents = {}
        for name in run_names:
            m = re.match(r"^(.+)_v(\d+)$", name)
            if m:
                base, num = m.group(1), int(m.group(2))
                candidate = f"{base}_v{num - 1}"
                if candidate in name_set:
                    parents[name] = candidate
        return parents

    @staticmethod
    def build_entry(name):
        """Tek run için tam entry dict oluştur."""
        params  = RunScanner.read_params(name)
        summary = RunScanner.read_summary(name)
        started, finished = RunScanner.get_timestamps(name)
        status = RunScanner.detect_status(name)

        # mode çıkarma
        # Run adı kesin kaynaktır: "baseline_*" → baseline, "fuzzy_*" → fuzzy
        # params.json'daki mode field eski bazı run'larda yanlış kaydedilmiş
        mode = None
        if name.startswith("baseline"):
            mode = "baseline"
        elif name.startswith("fuzzy"):
            mode = "fuzzy"
        elif name.startswith("debug"):
            mode = "debug"
        elif params:
            mode = params.get("mode")

        # base_lr çıkarma
        base_lr = None
        if params:
            base_lr = params.get("base_lr")
            if base_lr is None and "cli_args" in params:
                base_lr = params["cli_args"].get("base_lr")
            if base_lr is None and "ultralytics_args" in params:
                base_lr = params["ultralytics_args"].get("lr0")

        # best_map50: summary'den yoksa csv'den
        best_map50 = None
        if summary and "best_map50" in summary:
            best_map50 = summary["best_map50"]
        if best_map50 is None:
            best_map50 = RunScanner.recover_map50(name)

        # total_epochs / total_steps
        total_epochs = summary.get("total_epochs") if summary else None
        total_steps  = summary.get("total_steps")  if summary else None

        return {
            "name":         name,
            "started_at":   started,
            "finished_at":  finished,
            "status":       status,
            "parent":       None,       # migrate sonrası doldurulur
            "notes":        [],
            "mode":         mode,
            "base_lr":      base_lr,
            "best_map50":   best_map50,
            "total_epochs": total_epochs,
            "total_steps":  total_steps,
        }

# ============================================================================
#  RunRegistry  —  run_history.json yönetimi
# ============================================================================
class RunRegistry:

    def __init__(self):
        self.history_file = HISTORY_FILE

    # --- load / save -----------------------------------------------------------
    def load(self):
        if not os.path.isfile(self.history_file):
            return {"version": 1, "entries": []}
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"version": 1, "entries": []}

    def save(self, data):
        # Atomic write: temp dosyaya yaz, sonra rename
        tmp = self.history_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.history_file)

    # --- helpers ---------------------------------------------------------------
    def _find_entry(self, data, name):
        """Entries listesinde name'e göre entry bul, (index, entry) dön."""
        for i, e in enumerate(data["entries"]):
            if e["name"] == name:
                return i, e
        return None, None

    # --- public ----------------------------------------------------------------
    def register_run(self, name, started_at=None):
        """Yeni run kaydet. Zaten var ise sadece started_at güncelle."""
        data = self.load()
        idx, existing = self._find_entry(data, name)

        if existing is not None:
            # Zaten var — started_at'ı güncelle (yeniden çalıştırıldı)
            if started_at is None:
                started_at = datetime.now().isoformat(timespec="seconds")
            existing["started_at"] = started_at
            existing["status"] = "running"
            existing["finished_at"] = None
        else:
            # Yeni entry
            if started_at is None:
                started_at = datetime.now().isoformat(timespec="seconds")
            data["entries"].append({
                "name":         name,
                "started_at":   started_at,
                "finished_at":  None,
                "status":       "running",
                "parent":       None,
                "notes":        [],
                "mode":         None,
                "base_lr":      None,
                "best_map50":   None,
                "total_epochs": None,
                "total_steps":  None,
            })

        self.save(data)

    def add_note(self, name, text):
        data = self.load()
        idx, entry = self._find_entry(data, name)
        if entry is None:
            return False
        entry["notes"].append({
            "text":     text,
            "added_at": datetime.now().isoformat(timespec="seconds"),
        })
        self.save(data)
        return True

    def set_parent(self, name, parent_name):
        data = self.load()
        idx, entry = self._find_entry(data, name)
        if entry is None:
            return False
        # parent'ın var olduğunu kontrol et
        _, parent_entry = self._find_entry(data, parent_name)
        if parent_entry is None:
            return False
        entry["parent"] = parent_name
        self.save(data)
        return True

# ============================================================================
#  RunFormatter  —  terminal çıktı formatları
# ============================================================================
class RunFormatter:

    @staticmethod
    def _lr_str(lr):
        if lr is None:
            return "--"
        return f"{lr:.5f}"

    @staticmethod
    def _map_str(m):
        if m is None:
            return "--"
        return f"{m:.4f}"

    @staticmethod
    def format_list(entries):
        """Tablo formatında tüm run'lar."""
        header = (
            f" {'#':>3}  {'Run Name':<32} {'Started':<18} "
            f"{'Status':<9} {'Mode':<9} {'LR':<10} {'mAP50'}"
        )
        sep = "-" * len(header)
        lines = [
            "=== DL-DETECTION RUN HISTORY ===",
            header,
            sep,
        ]
        for i, e in enumerate(entries, 1):
            started = (e.get("started_at") or "--")[:16]
            lines.append(
                f" {i:>3}  {e['name']:<32} {started:<18} "
                f"{e.get('status', '--'):<9} "
                f"{(e.get('mode') or '--'):<9} "
                f"{RunFormatter._lr_str(e.get('base_lr')):<10} "
                f"{RunFormatter._map_str(e.get('best_map50'))}"
            )
        return "\n".join(lines)

    @staticmethod
    def format_show(entry, parent_entry=None):
        """Tek run detay görünümü."""
        lines = [
            f"=== {entry['name']} ===",
            f"Status      : {entry.get('status', '--')}",
            f"Started     : {entry.get('started_at') or '--'}",
            f"Finished    : {entry.get('finished_at') or '--'}",
        ]

        # Parent satırı
        parent_name = entry.get("parent")
        if parent_name:
            p_status = parent_entry.get("status", "?") if parent_entry else "?"
            lines.append(f"Parent      : {parent_name}  ({p_status})")
        else:
            lines.append("Parent      : --")

        lines += [
            f"Mode        : {entry.get('mode') or '--'}",
            f"Base LR     : {RunFormatter._lr_str(entry.get('base_lr'))}",
            f"Best mAP50  : {RunFormatter._map_str(entry.get('best_map50'))}",
            f"Total Epochs: {entry.get('total_epochs') or '--'}",
            f"Total Steps : {entry.get('total_steps') or '--'}",
        ]

        # Notes
        notes = entry.get("notes", [])
        if notes:
            lines.append("")
            lines.append("Notes       :")
            for n in notes:
                lines.append(f"  [{n.get('added_at', '?')[:10]}] {n['text']}")
        else:
            lines.append("Notes       : (yok)")

        return "\n".join(lines)

    @staticmethod
    def format_last(entries, n=1):
        """Son N run özeti."""
        recent = entries[-n:] if n <= len(entries) else entries
        lines = [f"=== SON {len(recent)} RUN ===", ""]
        for e in reversed(recent):
            lines += [
                f"  {e['name']}",
                f"    Started   : {e.get('started_at') or '--'}",
                f"    Status    : {e.get('status', '--')}",
                f"    Mode / LR : {e.get('mode') or '--'} / {RunFormatter._lr_str(e.get('base_lr'))}",
                f"    Best mAP50: {RunFormatter._map_str(e.get('best_map50'))}",
                "",
            ]
        return "\n".join(lines)

# ============================================================================
#  CLI KOMUTLARI
# ============================================================================

def cmd_migrate(args):
    """Mevcut run klasörlerini run_history.json'a ekle (bir kere)."""
    if os.path.isfile(HISTORY_FILE):
        print("[MIGRATE] run_history.json zaten var. Üzerine yazılmayacak.")
        print("[MIGRATE] Eğer yeniden migrate etmek istemek istersen dosyayı el ile sil.")
        return

    run_names = RunScanner.scan_all()
    if not run_names:
        print("[MIGRATE] artifacts/runs/ boş veya bulunamadı.")
        return

    print(f"[MIGRATE] {len(run_names)} run klasörü bulundu. Taranıyor...")
    entries = []
    for name in run_names:
        entry = RunScanner.build_entry(name)
        status = entry["status"]
        map_s  = RunFormatter._map_str(entry["best_map50"])
        print(f"  {name:<36} | {status:<9} mAP50={map_s}")
        entries.append(entry)

    # Parent auto-detect
    parents = RunScanner.detect_parents(run_names)
    if parents:
        print("\n[MIGRATE] Auto-detected parents:")
        for child, parent in parents.items():
            print(f"  {child} -> {parent}")
            for e in entries:
                if e["name"] == child:
                    e["parent"] = parent

    # Tarihe göre sıra (started_at ile)
    entries.sort(key=lambda e: e.get("started_at") or "")

    data = {"version": 1, "entries": entries}
    registry = RunRegistry()
    registry.save(data)
    print(f"\n[MIGRATE] run_history.json kayıt edildi. ({len(entries)} entry)")
    print("[MIGRATE] Not eklemek için:")
    print('  python train/run_tracker.py note <run_name> "<not metin>"')


def cmd_list(args):
    """Tüm run'ları listele (filtreleme destekli)."""
    registry = RunRegistry()
    data = registry.load()
    entries = data.get("entries", [])

    if not entries:
        print("[INFO] run_history.json boş veya yok. Önce 'migrate' komutunu çalıştır.")
        return

    # Filtre
    if args.mode:
        entries = [e for e in entries if e.get("mode") == args.mode]
    if args.status:
        entries = [e for e in entries if e.get("status") == args.status]

    if not entries:
        print("[INFO] Filtre sonucu boş.")
        return

    print(RunFormatter.format_list(entries))


def cmd_last(args):
    """Son N run özeti."""
    registry = RunRegistry()
    data = registry.load()
    entries = data.get("entries", [])

    if not entries:
        print("[INFO] run_history.json boş veya yok. Önce 'migrate' komutunu çalıştır.")
        return

    print(RunFormatter.format_last(entries, args.n))


def cmd_show(args):
    """Tek run detay."""
    registry = RunRegistry()
    data = registry.load()
    entries = data.get("entries", [])

    _, entry = registry._find_entry(data, args.name)
    if entry is None:
        print(f"[ERROR] '{args.name}' bulunamadı.")
        return

    # Parent entry (varsa)
    parent_entry = None
    if entry.get("parent"):
        _, parent_entry = registry._find_entry(data, entry["parent"])

    print(RunFormatter.format_show(entry, parent_entry))


def cmd_note(args):
    """Run'a not ekle."""
    registry = RunRegistry()
    data = registry.load()
    _, entry = registry._find_entry(data, args.name)
    if entry is None:
        print(f"[ERROR] '{args.name}' bulunamadı.")
        return

    if not registry.add_note(args.name, args.text):
        print(f"[ERROR] Not eklenemedi.")
        return
    print(f"[OK] Not added to {args.name}")


def cmd_parent(args):
    """Parent ilişkisi ata."""
    registry = RunRegistry()
    data = registry.load()

    _, child = registry._find_entry(data, args.name)
    if child is None:
        print(f"[ERROR] '{args.name}' bulunamadı.")
        return

    _, parent = registry._find_entry(data, args.parent_name)
    if parent is None:
        print(f"[ERROR] Parent '{args.parent_name}' bulunamadı.")
        return

    if not registry.set_parent(args.name, args.parent_name):
        print("[ERROR] Parent atanamadı.")
        return
    print(f"[OK] {args.name} -> parent: {args.parent_name}")


# ============================================================================
#  MAIN  —  argparse CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="dl-detection run history tracker",
        epilog="Örnek: python train/run_tracker.py list --mode fuzzy"
    )
    sub = ap.add_subparsers(dest="command", help="Komut")

    # migrate
    sub.add_parser("migrate", help="Mevcut run'ları history'ye ekle (bir kere)")

    # list
    p_list = sub.add_parser("list", help="Tüm run'ları listele")
    p_list.add_argument("--mode",   choices=["fuzzy", "baseline"], help="Mode filtresi")
    p_list.add_argument("--status", choices=["complete", "crashed", "running", "unknown"],
                        help="Status filtresi")

    # last
    p_last = sub.add_parser("last", help="Son N run özeti")
    p_last.add_argument("n", nargs="?", type=int, default=1, help="Son kaç run (default: 1)")

    # show
    p_show = sub.add_parser("show", help="Tek run detay")
    p_show.add_argument("name", help="Run adı")

    # note
    p_note = sub.add_parser("note", help="Run'a not ekle")
    p_note.add_argument("name", help="Run adı")
    p_note.add_argument("text", help="Not metni")

    # parent
    p_par = sub.add_parser("parent", help="Parent ilişkisi ata")
    p_par.add_argument("name",        help="Child run adı")
    p_par.add_argument("parent_name", help="Parent run adı")

    args = ap.parse_args()

    if args.command is None:
        ap.print_help()
        return

    dispatch = {
        "migrate": cmd_migrate,
        "list":    cmd_list,
        "last":    cmd_last,
        "show":    cmd_show,
        "note":    cmd_note,
        "parent":  cmd_parent,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
