# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 配置: onedir + 无控制台
# 构建: pyinstaller plantroot.spec   -> 产物在 dist/PlantRootRecon/

a = Analysis(
    ['desktop_app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('templates/index.html', 'templates'),
    ],
    hiddenimports=[
        'web_core_matlab',
        'web_core_images',
        'web_core_genviews',
        'matplotlib.backends.backend_agg',
        'pywebview.platforms.winforms',
        'pywebview.platforms.edgechromium',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PlantRootRecon',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='PlantRootRecon',
)
