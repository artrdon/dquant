import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from cycler import cycler
import os


base_config = {
    # ОСНОВНЫЕ ПАРАМЕТРЫ
    'figure': {
        'figsize': (12, 6),
        'dpi': 100,
        'edgecolor': 'white',
        'tight_layout': True,
        'window_title': 'dquant'
    },

    # ШРИФТЫ
    'font': {
        'family': 'DejaVu Sans',
        'size': 11,
        'weight': 'normal',
        'title_size': 14,
        'title_weight': 'bold',
        'label_size': 12,
        'label_weight': 'bold'
    },

    # ЦВЕТА ЛИНИЙ (общие для обеих тем)
    'line_cycle': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFE194', '#BAA6DD'],

    # ЛИНИИ И МАРКЕРЫ
    'lines': {
        'linewidth': 2.5,
        'linestyle': '-',
        'markersize': 6,
        'marker': 'o',
        'markevery': 10,
        'alpha': 1.0
    },

    # ЛЕГЕНДА
    'legend': {
        'loc': 'best',
        'fontsize': 10,
        'framealpha': 0.9,
        'fancybox': True,
        'shadow': True,
        'borderpad': 0.5,
        'labelcolor': 'text.color'
    },

    # ОСИ
    'axes': {
        'spines_visible': True,
        'spines_linewidth': 1,
        'tick_direction': 'in',
        'tick_length': 6,
        'tick_width': 1
    },

    # СОХРАНЕНИЕ
    'save': {
        'format': 'png',
        'dpi': 300,
        'bbox_inches': 'tight',
        'pad_inches': 0.1,
        'transparent': False,
        'facecolor': 'auto'
    }
}

# Светлая тема
light_theme = {
    'name': 'light',
    'figure': {
        'facecolor': '#f8f9fa',
    },
    'colors': {
        'primary': '#FF6B6B',
        'secondary': '#4ECDC4',
        'tertiary': '#45B7D1',
        'quaternary': '#96CEB4',
        'background': '#f8f9fa',
        'grid': '#dddddd',
        'text': '#2c3e50',
        'spines': '#2c3e50',
        'brand': '#2c3e50',
        'legend_bg': '#ffffff',
        'legend_edge': '#cccccc'
    },
    'grid': {
        'visible': True,
        'alpha': 0.3,
        'linestyle': '--',
        'linewidth': 0.5,
        'color': '#dddddd',
        'which': 'both'
    }
}

# Темная тема
dark_theme = {
    'name': 'dark',
    'figure': {
        'facecolor': '#1a2634',
    },
    'colors': {
        'primary': '#FF6B6B',
        'secondary': '#4ECDC4',
        'tertiary': '#45B7D1',
        'quaternary': '#96CEB4',
        'background': '#2c3e50',
        'grid': '#4a5b6e',
        'text': '#ecf0f1',  # Белый текст
        'spines': '#ecf0f1',  # Белые границы
        'brand': '#ecf0f1',  # Белый бренд
        'legend_bg': '#2c3e50',
        'legend_edge': '#4a5b6e'
    },
    'grid': {
        'visible': True,
        'alpha': 0.5,
        'linestyle': '--',
        'linewidth': 0.5,
        'color': '#4a5b6e',
        'which': 'both'
    }
}


def create_config(theme='light', logo_path=None):
    """Создает полную конфигурацию на основе выбранной темы"""

    # Выбираем тему
    if theme == 'dark':
        theme_config = dark_theme
    else:
        theme_config = light_theme

    # Объединяем с базовой конфигурацией
    config = base_config.copy()
    config['theme'] = theme_config['name']

    # Добавляем параметры фигуры из темы
    config['figure']['facecolor'] = theme_config['figure']['facecolor']

    # Добавляем цвета из темы
    config['colors'] = theme_config['colors'].copy()
    config['colors']['line_cycle'] = base_config['line_cycle']

    # Добавляем сетку из темы
    config['grid'] = theme_config['grid'].copy()

    # Добавляем параметры осей из темы
    config['axes']['spines_color'] = theme_config['colors']['spines']

    # Добавляем бренд
    config['brand'] = {
        'name': 'Dvol',
        'fontsize': 16,
        'fontweight': 'bold',
        'color': theme_config['colors']['brand'],
        'position': (0.02, 0.96),
        'use_logo': logo_path is not None and os.path.exists(logo_path),
        'logo_path': logo_path,
        'logo_size': 0.08
    }

    return config


class Visualization:

    def __init__(self, theme='light'):
        self.config = create_config(theme)
        self.theme = theme
        self._apply_config()

    def _apply_config(self):
        """Применяет конфигурацию к глобальным настройкам matplotlib"""
        config = self.config

        # Настройка шрифтов
        plt.rcParams['font.family'] = config['font']['family']
        plt.rcParams['font.size'] = config['font']['size']
        plt.rcParams['font.weight'] = config['font']['weight']
        plt.rcParams['axes.titlesize'] = config['font']['title_size']
        plt.rcParams['axes.titleweight'] = config['font']['title_weight']
        plt.rcParams['axes.labelsize'] = config['font']['label_size']
        plt.rcParams['axes.labelweight'] = config['font']['label_weight']

        # Настройка цветов
        plt.rcParams['axes.prop_cycle'] = cycler(color=config['colors']['line_cycle'])
        plt.rcParams['figure.facecolor'] = config['figure']['facecolor']
        plt.rcParams['axes.facecolor'] = config['colors']['background']
        plt.rcParams['text.color'] = config['colors']['text']

        # Настройка линий
        plt.rcParams['lines.linewidth'] = config['lines']['linewidth']
        plt.rcParams['lines.linestyle'] = config['lines']['linestyle']
        plt.rcParams['lines.markersize'] = config['lines']['markersize']
        plt.rcParams['lines.marker'] = config['lines']['marker']

        # Настройка сетки
        plt.rcParams['grid.alpha'] = config['grid']['alpha']
        plt.rcParams['grid.linestyle'] = config['grid']['linestyle']
        plt.rcParams['grid.linewidth'] = config['grid']['linewidth']
        plt.rcParams['grid.color'] = config['grid']['color']

        # Настройка осей
        plt.rcParams['axes.linewidth'] = config['axes']['spines_linewidth']
        plt.rcParams['axes.edgecolor'] = config['axes']['spines_color']
        plt.rcParams['xtick.direction'] = config['axes']['tick_direction']
        plt.rcParams['ytick.direction'] = config['axes']['tick_direction']
        plt.rcParams['xtick.major.size'] = config['axes']['tick_length']
        plt.rcParams['ytick.major.size'] = config['axes']['tick_length']
        plt.rcParams['xtick.color'] = config['colors']['text']
        plt.rcParams['ytick.color'] = config['colors']['text']

        # Явно устанавливаем цвета для label
        plt.rcParams['axes.labelcolor'] = config['colors']['text']
        plt.rcParams['axes.titlecolor'] = config['colors']['text']

        # Настройка легенды
        plt.rcParams['legend.facecolor'] = config['colors']['legend_bg']
        plt.rcParams['legend.edgecolor'] = config['colors']['legend_edge']
        plt.rcParams['legend.fontsize'] = config['legend']['fontsize']
        plt.rcParams['legend.framealpha'] = config['legend']['framealpha']
        plt.rcParams['legend.fancybox'] = config['legend']['fancybox']
        plt.rcParams['legend.shadow'] = config['legend']['shadow']
        plt.rcParams['legend.borderpad'] = config['legend']['borderpad']

        # Цвет текста в легенде
        plt.rcParams['legend.labelcolor'] = config['colors']['text']

        # Применяем конфигурацию фигуры по умолчанию
        plt.rcParams['figure.figsize'] = config['figure']['figsize']
        plt.rcParams['figure.dpi'] = config['figure']['dpi']
        plt.rcParams['savefig.facecolor'] = config['figure']['facecolor']
        plt.rcParams['figure.titlesize'] = config['font']['title_size']
        plt.rcParams['figure.titleweight'] = config['font']['title_weight']

        # НЕ отключаем панель инструментов
        plt.rcParams['toolbar'] = 'toolbar2'

    def set_theme(self, theme):
        """Переключает тему"""
        self.config = create_config(theme)
        self.theme = theme
        self._apply_config()
        print(f"✅ Тема переключена на '{theme}'")

    def _create_figure(self, *args, **kwargs):
        """Создает фигуру с правильным фоном и заголовком окна"""
        fig, ax = plt.subplots(*args, **kwargs)

        # Устанавливаем заголовок окна
        try:
            fig.canvas.manager.set_window_title(self.config['figure']['window_title'])
        except:
            pass

        # Устанавливаем цвет фона для всей фигуры
        fig.patch.set_facecolor(self.config['figure']['facecolor'])
        fig.patch.set_alpha(1.0)

        return fig, ax

    def _style_axes(self, ax):
        """Применяет стилизацию к осям"""
        for spine in ax.spines.values():
            spine.set_color(self.config['colors']['spines'])
            spine.set_linewidth(self.config['axes']['spines_linewidth'])

        ax.tick_params(colors=self.config['colors']['text'],
                       direction=self.config['axes']['tick_direction'],
                       length=self.config['axes']['tick_length'],
                       width=self.config['axes']['tick_width'])

        # Явно устанавливаем цвет для label
        ax.xaxis.label.set_color(self.config['colors']['text'])
        ax.yaxis.label.set_color(self.config['colors']['text'])
        ax.title.set_color(self.config['colors']['text'])

    def _style_legend(self, ax):
        """Применяет стилизацию к легенде"""
        legend = ax.legend()
        if legend:
            legend.get_frame().set_facecolor(self.config['colors']['legend_bg'])
            legend.get_frame().set_edgecolor(self.config['colors']['legend_edge'])
            for text in legend.get_texts():
                text.set_color(self.config['colors']['text'])

    def save_figure(self, fig, filename):
        """Сохраняет фигуру с настройками из конфига"""
        config = self.config['save']

        save_kwargs = {
            'format': config['format'],
            'dpi': config['dpi'],
            'bbox_inches': config['bbox_inches'],
            'pad_inches': config['pad_inches'],
            'transparent': config['transparent'],
            'facecolor': self.config['figure']['facecolor']
        }

        fig.savefig(filename, **save_kwargs)
        print(f"💾 График сохранен как {filename}")

    def show_vol(self, df, pred, save_path=None):
        fig, ax = self._create_figure(figsize=(15, 6))

        bottom_y = 0
        max_bar_height = 0

        for i, (idx, row) in enumerate(df.iterrows()):
            if i < len(df)-pred:
                color = self.config['colors']['primary']
            else:
                color = 'green'

            x_pos = i


            ax.plot([x_pos, x_pos], [bottom_y, bottom_y],
                    color=self.config['colors']['text'], linewidth=1)

            bar_height = row['value']
            if bar_height > max_bar_height:
                max_bar_height = bar_height

            rect = patches.Rectangle((x_pos - 0.3, 0), 0.6, bar_height,
                                     linewidth=1, edgecolor=color, facecolor=color, alpha=0.7)
            ax.add_patch(rect)

        ax.set_xlim(-1, len(df))
        ax.set_ylim(0, max_bar_height * 1.3)
        ax.set_xlabel('Время')
        ax.set_ylabel('')
        ax.set_title('Объемный график')

        self._style_axes(ax)

        plt.xticks(range(len(df)), [d.strftime('%m-%d') for d in df.index], rotation=45)

        plt.tight_layout()

        if save_path:
            self.save_figure(fig, save_path)

        plt.show()

    def show_errors(self, train_errors, val_errors, save_path=None):
        fig, ax = self._create_figure(figsize=(10, 6))

        ax.plot(list(train_errors), label='Train Loss',
                color=self.config['colors']['primary'])
        ax.plot(list(val_errors), label='Validation Loss',
                color=self.config['colors']['secondary'])
        ax.set_xlabel('Trees')
        ax.set_ylabel('MSE Loss')
        ax.set_title('Training and Test Loss over Trees')
        ax.grid(True)

        self._style_axes(ax)
        self._style_legend(ax)

        plt.tight_layout()

        if save_path:
            self.save_figure(fig, save_path)

        plt.show()

    def test(self, save_path=None):
        x = np.linspace(0, 10, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)
        y3 = np.sin(x) * np.cos(x)

        fig, ax = self._create_figure(figsize=(12, 6))

        ax.plot(x, y1, label='sin(x)', color=self.config['colors']['primary'])
        ax.plot(x, y2, label='cos(x)', color=self.config['colors']['secondary'])
        ax.plot(x, y3, label='sin(x)cos(x)', color=self.config['colors']['tertiary'])

        ax.set_title('Тригонометрические функции')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)

        self._style_axes(ax)
        self._style_legend(ax)

        plt.tight_layout()

        if save_path:
            self.save_figure(fig, save_path)

        plt.show()


if __name__ == "__main__":
    v = Visualization(theme='dark')

    v.test(save_path='test_with_logo.png')