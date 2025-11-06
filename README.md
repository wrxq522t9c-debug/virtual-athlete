# virtual_athlete_final.py
import pygame, sys, csv, os
from collections import deque

pygame.init()

# ---------------- 字体加载（优先本地 CN_FONT.ttf） ----------------
FONT_FILE = "CN_FONT.ttf"   # 若不是放在同一目录，请改为绝对路径
FONT_SIZE = 16
if os.path.exists(FONT_FILE):
    try:
        FONT = pygame.font.Font(FONT_FILE, FONT_SIZE)
        print(f"✅ 已加载本地字体文件: {FONT_FILE}")
    except Exception as e:
        print(f"❌ 加载本地字体失败: {e}")
        FONT = None
else:
    print(f"⚠️ 本地字体文件未找到: {FONT_FILE} (cwd: {os.getcwd()})")
    FONT = None

if FONT is None:
    # 回退到常见系统字体
    for name in ("PingFang SC", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        try:
            FONT = pygame.font.SysFont(name, FONT_SIZE)
            print(f"ℹ️ 回退使用系统字体: {name}")
            break
        except Exception:
            FONT = None
if FONT is None:
    FONT = pygame.font.Font(None, FONT_SIZE)
    print("⚠️ 最终回退到默认字体（可能无法正确显示中文）")

# ---------------- 基础仿真参数 ----------------
BASE_DT = 0.05  # 每次 step 的基本时间步（秒）
# sim_speed_factor 由 UI 设置：0.5, 1.0, 2.0, 4.0

# ---------------- Athlete 类 ----------------
class Athlete:
    def __init__(self):
        self.params = {
            "体重(kg)": 70.0,
            "肌肉比例": 0.40,
            "快肌比例": 0.45,
            "最大摄氧量(VO2max)": 1.0,
            "最大功率(Pmax)": 1.0,
            "临界功率(CP)": 0.6,
            "无氧储备(W')": 0.2,
            "运动时间(min)": 10.0,
            "运动类型": "节奏跑",
            "间歇-工作强度": 0.95,
            "间歇-工作时间(s)": 30.0,
            "间歇-休息强度": 0.55,
            "间歇-休息时间(s)": 60.0,
        }
        self.modes = {
            "恢复跑": 0.5,
            "轻松跑": 0.65,
            "节奏跑": 0.8,
            "阈值跑": 0.88,
            "间歇跑": None,
            "冲刺": 1.0,
        }

    def set(self, key, val):
        self.params[key] = val

    def get(self, key):
        return self.params[key]

# ---------------- EnergyModel 类（接受 dt 参数） ----------------
class EnergyModel:
    def __init__(self, athlete: Athlete):
        self.a = athlete
        self.reset()

    def reset(self):
        self.t = 0.0
        self.ATP = 1.0
        self.PCr = 1.0
        self.VO2_fast = 0.0
        self.VO2_slow = 0.0
        self.VO2_total = 0.0
        self.Lac = 0.0
        self.Wrem = self.a.get("无氧储备(W')")
        self.power = 0.0
        self.P_target = 0.0
        self.interval_index = 0
        self.interval_elapsed = 0.0

    def get_target_power(self):
        mode = self.a.get("运动类型")
        if mode != "间歇跑":
            return self.a.modes[mode]
        # 间歇模式读取运动员参数
        wI = self.a.get("间歇-工作强度")
        wT = self.a.get("间歇-工作时间(s)")
        rI = self.a.get("间歇-休息强度")
        rT = self.a.get("间歇-休息时间(s)")
        pattern = [(wT, wI), (rT, rI)]
        dur, inten = pattern[self.interval_index]
        return inten, dur

    def step(self, dt):
        CP = self.a.get("临界功率(CP)")
        # 处理间歇目标（需要用 dt 更新 interval_elapsed）
        if self.a.get("运动类型") == "间歇跑":
            inten, dur = self.get_target_power()
            # get_target_power returned tuple; set P_target accordingly
            self.P_target = inten
            if self.interval_elapsed >= dur:
                self.interval_index = (self.interval_index + 1) % 2
                self.interval_elapsed = 0.0
                inten, dur = self.get_target_power()
                self.P_target = inten
            self.interval_elapsed += dt
        else:
            self.P_target = self.a.modes[self.a.get("运动类型")]

        # fatigue factor based on W'
        fatigue = self.Wrem / max(1e-6, self.a.get("无氧储备(W')"))
        fatigue = max(0.2, fatigue)
        P_eff = (self.P_target or 0.0) * fatigue
        # update actual power (first-order dynamics)
        self.power += (P_eff - self.power) / 3.0 * dt

        # VO2: fast and slow components
        VO2_fast_ss = min(1.0, self.power)
        VO2_slow_ss = 0.0
        if self.power > CP:
            VO2_slow_ss = min(0.2, (self.power - CP) * 0.5)
        self.VO2_fast += (VO2_fast_ss - self.VO2_fast) / 20.0 * dt
        self.VO2_slow += (VO2_slow_ss - self.VO2_slow) / 300.0 * dt
        self.VO2_total = self.VO2_fast + self.VO2_slow

        # PCr dynamics (recovery depends on VO2_total)
        k_depl = 0.02 + 0.5 * max(0.0, self.power - 0.6)
        self.PCr += (-k_depl * self.power * self.PCr + (1 - self.PCr) * self.VO2_total / 45.0) * dt

        # Lactate production & clearance
        self.Lac += (1.5 * max(0.0, self.power - CP) * (1.0 - self.PCr) - 0.0017 * self.Lac) * dt

        # ATP steady-state model (resynthesis depends on PCr and VO2_total)
        k_resyn = 0.12 * (self.PCr + self.VO2_total)
        k_use = 0.06 * self.power
        self.ATP += (k_resyn * (1.0 - self.ATP) - k_use * self.ATP) * dt

        # W' depletion / recovery
        if self.power > CP:
            self.Wrem -= (self.power - CP) * 0.01 * dt  # scaled by dt
        else:
            # recovery scaled by dt and VO2 effect implicitly via Wrem difference
            self.Wrem += (self.a.get("无氧储备(W')") - self.Wrem) / 300.0 * dt

        # bounds
        self.PCr = min(max(self.PCr, 0.0), 1.0)
        self.ATP = min(max(self.ATP, 0.0), 1.0)
        self.Wrem = min(max(self.Wrem, 0.0), self.a.get("无氧储备(W')"))
        self.Lac = max(self.Lac, 0.0)
        self.VO2_fast = min(max(self.VO2_fast, 0.0), 1.0)
        self.VO2_slow = min(max(self.VO2_slow, 0.0), 1.0)
        self.VO2_total = min(max(self.VO2_total, 0.0), 2.0)  # allow >1 if slow comp adds
        self.t += dt

# ---------------- SimulationApp（UI） ----------------
class SimulationApp:
    def __init__(self):
        # window
        self.WIDTH, self.HEIGHT = 1200, 700
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("虚拟运动员：三大供能系统动态变化")
        self.clock = pygame.time.Clock()

        # model
        self.athlete = Athlete()
        self.model = EnergyModel(self.athlete)

        # editing & UI states
        self.edit_keys = list(self.athlete.params.keys())
        self.edit_index = 0
        self.editing = False
        self.edit_text = ""
        self.show_mode_menu = False
        self.running_sim = False

        # left panel layout
        self.LEFT_W = 320
        # plot rect
        self.PLOT_RECT = pygame.Rect(self.LEFT_W + 10, 10, self.WIDTH - self.LEFT_W - 30, 400)

        # buffers
        self.buf_time = deque(maxlen=6000)
        self.buf_P = deque(maxlen=6000)
        self.buf_V = deque(maxlen=6000)
        self.buf_PCr = deque(maxlen=6000)
        self.buf_Lac = deque(maxlen=6000)
        self.buf_ATP = deque(maxlen=6000)

        # sim speed control (A option): clickable buttons 0.5x / 1x / 2x / 4x
        self.sim_speed_options = [0.5, 1.0, 2.0, 4.0]
        self.sim_speed_index = 1  # default points to 1.0
        self.sim_speed = self.sim_speed_options[self.sim_speed_index]

        # numeric editable keys set
        self.numeric_keys = {k for k, v in self.athlete.params.items() if isinstance(v, float)}

    # helper text draw
    def text(self, x, y, s, c=(0, 0, 0)):
        surf = FONT.render(str(s), True, c)
        self.screen.blit(surf, (int(x), int(y)))

    # buffers update
    def update_buffers(self):
        self.buf_time.append(self.model.t)
        self.buf_P.append(self.model.power)
        self.buf_V.append(self.model.VO2_total)
        self.buf_PCr.append(self.model.PCr)
        self.buf_Lac.append(self.model.Lac)
        self.buf_ATP.append(self.model.ATP)

    def clear_buffers(self):
        self.buf_time.clear()
        self.buf_P.clear()
        self.buf_V.clear()
        self.buf_PCr.clear()
        self.buf_Lac.clear()
        self.buf_ATP.clear()

    def export_to_csv(self):
        if not self.buf_time:
            print("❌ 无数据可导出")
            return
        name = f"sim_data_{pygame.time.get_ticks()}.csv"
        with open(name, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t(s)", "Power", "VO2_total", "PCr", "Lactate", "ATP"])
            for row in zip(self.buf_time, self.buf_P, self.buf_V, self.buf_PCr, self.buf_Lac, self.buf_ATP):
                w.writerow(row)
        print("✅ 已导出:", name)

    # ---------- event handling ----------
    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos

                # 1) if mode menu open, check selection first (menu position adaptive)
                if self.show_mode_menu:
                    idx = self.edit_keys.index("运动类型")
                    base_y = 40 + idx * 25
                    options = list(self.athlete.modes.keys())
                    for j, opt in enumerate(options):
                        rect = pygame.Rect(self.LEFT_W - 120, base_y + j * 25, 120, 22)
                        if rect.collidepoint(mx, my):
                            self.athlete.set("运动类型", opt)
                            self.show_mode_menu = False
                            self.model.reset()
                            self.clear_buffers()
                            return
                    # click outside menu: close
                    self.show_mode_menu = False

                # 2) left panel row clicks
                y = 40
                for i, key in enumerate(self.edit_keys):
                    row_rect = pygame.Rect(10, y, self.LEFT_W - 20, 22)
                    if row_rect.collidepoint(mx, my):
                        self.edit_index = i
                        if key == "运动类型":
                            # toggle menu
                            self.show_mode_menu = not self.show_mode_menu
                        else:
                            # if editing numeric allowed
                            if key in self.numeric_keys:
                                self.editing = True
                                self.edit_text = ""
                            else:
                                # non-numeric fields can't be edited by keyboard here
                                self.editing = False
                        return
                    y += 25

                # 3) left buttons (Play/Pause/Reset/Export/Sim Speed)
                btn_y = self.HEIGHT - 220
                # Play
                if 20 < mx < 140 and btn_y < my < btn_y + 40:
                    self.running_sim = True
                # Pause
                if 160 < mx < 280 and btn_y < my < btn_y + 40:
                    self.running_sim = False
                # Reset
                if 20 < mx < 280 and btn_y + 60 < my < btn_y + 100:
                    self.model.reset()
                    self.clear_buffers()
                # Export
                if 20 < mx < 280 and btn_y + 120 < my < btn_y + 160:
                    self.export_to_csv()
                # Sim speed buttons (placed above Play row)
                # draw layout will create rectangles starting at x=20 y=btn_y-90
                spx = 20; spy = btn_y - 90; spw = 60; sph = 28; gap = 10
                for idx, val in enumerate(self.sim_speed_options):
                    rx = spx + idx * (spw + gap)
                    if rx < mx < rx + spw and spy < my < spy + sph:
                        self.sim_speed_index = idx
                        self.sim_speed = self.sim_speed_options[self.sim_speed_index]
                        print(f"ℹ️ 仿真速度设置为 {self.sim_speed}x")
                        return

            # keyboard input for numeric editing
            if ev.type == pygame.KEYDOWN and self.editing:
                if ev.key == pygame.K_RETURN:
                    key = self.edit_keys[self.edit_index]
                    try:
                        val = float(self.edit_text)
                        # enforce positive for time fields
                        if "时间" in key and val <= 0:
                            val = 1.0
                        self.athlete.set(key, val)
                    except:
                        pass
                    self.editing = False
                elif ev.key == pygame.K_BACKSPACE:
                    self.edit_text = self.edit_text[:-1]
                elif ev.unicode and (ev.unicode.isdigit() or ev.unicode in ".-"):
                    self.edit_text += ev.unicode
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_UP:
                    self.edit_index = (self.edit_index - 1) % len(self.edit_keys)
                elif ev.key == pygame.K_DOWN:
                    self.edit_index = (self.edit_index + 1) % len(self.edit_keys)

    # ---------- drawing helpers ----------
    def draw_left_panel(self):
        pygame.draw.rect(self.screen, (240, 240, 240), (0, 0, self.LEFT_W, self.HEIGHT))
        self.text(10, 10, "虚拟运动员属性（可点击编辑）")
        y = 40
        for i, key in enumerate(self.edit_keys):
            v = self.athlete.get(key)
            # gray out interval params unless mode is interval
            is_interval_param = key.startswith("间歇-")
            interval_active = (self.athlete.get("运动类型") == "间歇跑")
            if is_interval_param and not interval_active:
                col = (140, 140, 140)
            else:
                col = (255, 0, 0) if i == self.edit_index else (0, 0, 0)

            # label and value positions fixed
            self.text(10, y, f"{key}:", col)
            if key == "运动类型":
                self.text(200, y, str(v), col)
                # draw dropdown symbol to the right
                sym_x = self.LEFT_W - 40
                sym = "▲" if self.show_mode_menu else "▼"
                self.text(sym_x, y, sym, (90, 90, 90))
                # menu drawn when show_mode_menu True
                if self.show_mode_menu:
                    options = list(self.athlete.modes.keys())
                    menu_x = self.LEFT_W - 120
                    for j, opt in enumerate(options):
                        rect = pygame.Rect(menu_x, y + j * 25, 120, 22)
                        pygame.draw.rect(self.screen, (230, 230, 255), rect)
                        self.text(menu_x + 5, y + j * 25 + 3, opt)
            else:
                if self.editing and i == self.edit_index:
                    # editing input field
                    display = self.edit_text + "_"
                    self.text(200, y, display, col)
                else:
                    # display numeric rounded or string
                    if isinstance(v, float):
                        self.text(200, y, f"{v:.3f}", col)
                    else:
                        self.text(200, y, str(v), col)
            y += 25

        # status values
        self.text(10, y + 10, f"t = {self.model.t:.1f}s")
        self.text(10, y + 30, f"Target P = {self.model.P_target:.2f}")
        self.text(10, y + 50, f"Actual P = {self.model.power:.2f}")
        self.text(10, y + 70, f"VO2 = {self.model.VO2_total:.2f}")
        self.text(10, y + 90, f"PCr = {self.model.PCr:.2f}")
        self.text(10, y + 110, f"Lac = {self.model.Lac:.2f}")
        self.text(10, y + 130, f"ATP = {self.model.ATP:.2f}")

        # sim speed controls (A) - small buttons
        btn_y = self.HEIGHT - 220
        spx = 20; spy = btn_y - 90; spw = 60; sph = 28; gap = 10
        self.text(spx, spy - 18, "仿真速度：")
        for idx, val in enumerate(self.sim_speed_options):
            rx = spx + idx * (spw + gap)
            rect = pygame.Rect(rx, spy, spw, sph)
            color = (180, 220, 255) if idx == self.sim_speed_index else (220, 220, 220)
            pygame.draw.rect(self.screen, color, rect)
            self.text(rx + 12, spy + 6, f"{val}x")

        # buttons: Play / Pause / Reset / Export
        pygame.draw.rect(self.screen, (200, 200, 255), (20, btn_y, 120, 40)); self.text(60, btn_y + 12, "Play")
        pygame.draw.rect(self.screen, (200, 200, 200), (160, btn_y, 120, 40)); self.text(195, btn_y + 12, "Pause")
        pygame.draw.rect(self.screen, (255, 200, 200), (20, btn_y + 60, 260, 40)); self.text(120, btn_y + 72, "Reset")
        pygame.draw.rect(self.screen, (220, 255, 220), (20, btn_y + 120, 260, 40)); self.text(110, btn_y + 132, "保存数据 (CSV)")

        # small hint text
        self.text(10, self.HEIGHT - 32, "说明：点击字段编辑（仅数值），运动类型请用下拉选择；设置好后点击 Play。", (80, 80, 80))

    # ---------- plot drawing ----------
    def draw_plot(self):
        rect = self.PLOT_RECT
        pygame.draw.rect(self.screen, (255, 255, 255), rect)
        pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
        x0, y0, w, h = rect

        # ATP 安全区 (0.8-1.0)
        y_high = int(y0 + h - int(1.0 * h))
        y_low = int(y0 + h - int(0.8 * h))
        pygame.draw.rect(self.screen, (255, 230, 230), (x0, y_high, w, y_low - y_high))
        self.text(x0 + 10, y_high - 16, "ATP 安全区 (0.8–1.0)", (160, 0, 0))

        if len(self.buf_time) > 1:
            tmin, tmax = self.buf_time[0], self.buf_time[-1]
            span = max(1e-3, tmax - tmin)
            def tx(t): return int(x0 + (t - tmin) / span * w)
            def sy(v): return int(y0 + h - v * h)

            # draw vertical key times if within span
            key_times = [(2, "ATP–PCr 主导"), (10, "糖酵解主导"), (60, "有氧糖酵解主导"), (7200, "脂肪氧化主导")]
            for kt, label in key_times:
                if tmin <= kt <= tmax:
                    xline = tx(kt)
                    pygame.draw.line(self.screen, (180, 180, 180), (xline, y0), (xline, y0 + h), 1)
                    self.text(xline - 40, y0 + 5, label, (100, 100, 100))

            # draw grid horizontal lines (0.2..1.0)
            for v in [0.2, 0.4, 0.6, 0.8, 1.0]:
                yline = sy(v)
                pygame.draw.line(self.screen, (240, 240, 240), (x0, yline), (x0 + w, yline), 1)
                self.text(x0 - 36, yline - 8, f"{v:.1f}", (80, 80, 80))

            # draw curves with integer pixel coordinates
            def draw_curve(buf, color, label, y_off=0):
                pts = [(tx(t), sy(v)) for t, v in zip(self.buf_time, buf)]
                if len(pts) > 1:
                    pygame.draw.lines(self.screen, color, False, pts, 2)
                    self.text(x0 + w - 160, y0 + 10 + y_off, label, color)

            draw_curve(self.buf_P, (200, 50, 200), "Power(强度)", 0)
            draw_curve(self.buf_V, (50, 50, 200), "VO2(有氧)", 20)
            draw_curve(self.buf_PCr, (230, 120, 20), "PCr(ATP-CP)", 40)
            draw_curve(self.buf_Lac, (20, 160, 60), "Lactate(无氧糖解)", 60)
            draw_curve(self.buf_ATP, (255, 0, 0), "ATP(剩余量)", 80)
        else:
            self.text(x0 + 10, y0 + 10, "（运行后显示曲线）", (120, 120, 120))

        # x-axis label and y-axis label
        self.text(x0 + w // 2 - 30, y0 + h + 6, "时间 (s)")
        self.text(x0 - 70, y0 + h // 2 - 20, "系统活跃程度")

    # ---------- bars drawing ----------
    def draw_bars(self):
        rect = pygame.Rect(self.LEFT_W + 10, self.PLOT_RECT.bottom + 15, self.WIDTH - self.LEFT_W - 30, self.HEIGHT - (self.PLOT_RECT.bottom + 25))
        pygame.draw.rect(self.screen, (245, 245, 245), rect)
        pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
        x, y, w, h = rect

        # compute contributions (scaled for visual)
        c_atp = max(0.1, self.model.PCr * 2.0)
        W_used = max(0.0, self.athlete.get("无氧储备(W')") - self.model.Wrem)
        c_anaer = W_used * 10.0
        c_lac = self.model.Lac
        c_aero = self.model.VO2_total * 1.5

        S = c_atp + c_anaer + c_lac + c_aero + 1e-6
        vals = [c_atp / S, c_anaer / S, c_lac / S, c_aero / S]
        labels = ["ATP-PCr 系统", "无氧糖酵解 (W')", "乳酸贡献", "总有氧系统"]
        colors = [(230, 120, 20), (200, 50, 200), (20, 160, 60), (50, 50, 200)]

        by = y + 20; bh = 36
        for i, v in enumerate(vals):
            if by + i * (bh + 10) + bh > y + h: break
            pygame.draw.rect(self.screen, (220, 220, 220), (x + 20, by + i * (bh + 10), w - 40, bh))
            pygame.draw.rect(self.screen, colors[i], (x + 20, by + i * (bh + 10), int((w - 40) * v), bh))
            self.text(x + 26, by + i * (bh + 10) + 8, f"{labels[i]}: {v * 100:.0f}%")

    # ---------- main draw ----------
    def draw(self):
        self.screen.fill((255, 255, 255))
        self.draw_left_panel()
        self.draw_plot()
        self.draw_bars()

    # ---------- main loop ----------
    def run(self):
        while True:
            self.handle_events()
            # when running, integrate with dt scaled by sim_speed
            if self.running_sim:
                dt_effective = BASE_DT * self.sim_speed
                # to keep numerical stability, perform smaller substeps if dt_effective large:
                # split into N substeps where each substep <= 0.1s
                max_sub = max(1, int((dt_effective / 0.05) + 0.5))
                sub_dt = dt_effective / max_sub
                for _ in range(max_sub):
                    self.model.step(sub_dt)
                self.update_buffers()
                # auto-stop when reached athlete duration (min -> sec)
                if self.model.t > self.athlete.get("运动时间(min)") * 60.0:
                    self.running_sim = False

            self.draw()
            pygame.display.flip()
            self.clock.tick(30)


# ---------------- run app ----------------
if __name__ == "__main__":
    app = SimulationApp()
    app.run()
