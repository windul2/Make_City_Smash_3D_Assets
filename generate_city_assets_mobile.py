#!/usr/bin/env python3
"""Generate small, texture-free glTF 2.0/GLB models for City Smash.

Only Python's standard library is required; glTF coordinates are Y-up in metres.
Each model is an independent mesh whose ground contact is at Y=0, except flying
objects (meteor and drone), whose origins are at their visual centres.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


# name: (base color, metallic, roughness). No textures or external resources.
MATERIALS = {
    "asphalt": ("#263945", 0.0, 0.98),
    "concrete": ("#687f87", 0.0, 0.9),
    "steel": ("#526b77", 0.35, 0.58),
    "dark_steel": ("#324959", 0.3, 0.65),
    "glass": ("#407d90", 0.22, 0.32),
    "glass_dark": ("#2b5266", 0.2, 0.4),
    "cyan": ("#65eadc", 0.1, 0.35),
    "warm": ("#f4cf83", 0.05, 0.55),
    "orange": ("#ee8254", 0.1, 0.55),
    "rust": ("#a16b56", 0.1, 0.85),
    "blue": ("#537eab", 0.15, 0.55),
    "cream": ("#a8ac9e", 0.0, 0.85),
    "green": ("#4e987b", 0.0, 0.94),
    "green_light": ("#70b889", 0.0, 0.9),
    "trunk": ("#775c4a", 0.0, 0.95),
    "red": ("#bf685d", 0.08, 0.68),
    "white": ("#d6e6e4", 0.05, 0.58),
    "meteor": ("#674940", 0.08, 0.93),
    "ember": ("#ff9f60", 0.0, 0.62),
    "water": ("#438c9d", 0.0, 0.42),
    "black": ("#1d2934", 0.0, 0.88),
}


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def subtract(a, b):
    return tuple(x - y for x, y in zip(a, b))


def scale(a, factor):
    return tuple(x * factor for x in a)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def unit(a):
    length = math.sqrt(sum(v * v for v in a))
    return scale(a, 1 / length) if length else (0.0, 1.0, 0.0)


class Mesh:
    def __init__(self):
        self.groups = {}

    def triangle(self, a, b, c, material):
        normal = unit(cross(subtract(b, a), subtract(c, a)))
        group = self.groups.setdefault(material, {"positions": [], "normals": []})
        group["positions"].extend((*a, *b, *c))
        group["normals"].extend((*normal, *normal, *normal))

    def quad(self, a, b, c, d, material):
        self.triangle(a, b, c, material)
        self.triangle(a, c, d, material)

    def box(self, x, y, z, w, h, d, material):
        x0, x1 = x - w / 2, x + w / 2
        y0, y1 = y, y + h
        z0, z1 = z - d / 2, z + d / 2
        faces = (
            ((x0,y0,z0),(x0,y1,z0),(x1,y1,z0),(x1,y0,z0)),
            ((x1,y0,z1),(x1,y1,z1),(x0,y1,z1),(x0,y0,z1)),
            ((x0,y0,z1),(x0,y1,z1),(x0,y1,z0),(x0,y0,z0)),
            ((x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)),
            ((x0,y1,z0),(x0,y1,z1),(x1,y1,z1),(x1,y1,z0)),
            ((x0,y0,z1),(x0,y0,z0),(x1,y0,z0),(x1,y0,z1)),
        )
        for face in faces:
            self.quad(*face, material)

    def cylinder(self, x, y, z, radius, height, material, sides=8, axis="y"):
        def point(theta, longitudinal):
            a, b = radius * math.cos(theta), radius * math.sin(theta)
            if axis == "x":
                return (x + longitudinal, y + b, z + a)
            return (x + a, y + longitudinal, z + b)

        near = (x, y, z)
        far = (x + height, y, z) if axis == "x" else (x, y + height, z)
        for i in range(sides):
            a, b = i * math.tau / sides, (i + 1) * math.tau / sides
            p0, p1, q0, q1 = point(a, 0), point(b, 0), point(a, height), point(b, height)
            self.quad(p0, q0, q1, p1, material)
            self.triangle(near, p1, p0, material)
            self.triangle(far, q0, q1, material)

    def cone(self, x, y, z, radius, height, material, sides=7):
        top = (x, y + height, z)
        for i in range(sides):
            a, b = i * math.tau / sides, (i + 1) * math.tau / sides
            p = (x + radius * math.cos(a), y, z + radius * math.sin(a))
            q = (x + radius * math.cos(b), y, z + radius * math.sin(b))
            self.triangle(p, top, q, material)
            self.triangle((x, y, z), q, p, material)

    def octahedron(self, x, y, z, rx, ry, rz, material):
        top, bottom = (x,y+ry,z), (x,y-ry,z)
        ring = ((x-rx,y,z),(x,y,z+rz),(x+rx,y,z),(x,y,z-rz))
        for i in range(4):
            a, b = ring[i], ring[(i + 1) % 4]
            self.triangle(top, a, b, material)
            self.triangle(bottom, b, a, material)

    def beam(self, start, end, width, material):
        direction = unit(subtract(end, start))
        reference = (1,0,0) if abs(direction[1]) > .9 else (0,1,0)
        a = scale(unit(cross(direction, reference)), width / 2)
        b = scale(unit(cross(direction, a)), width / 2)
        corners = []
        for point in (start, end):
            corners.extend((add(add(point,a),b), add(subtract(point,a),b),
                            subtract(subtract(point,a),b), add(subtract(point,b),a)))
        for indices in ((0,1,2,3),(4,7,6,5),(0,4,5,1),(1,5,6,2),
                        (2,6,7,3),(3,7,4,0)):
            self.quad(*(corners[k] for k in indices),material)

    def window_rows(self, x, z, w, d, height, rows, columns=2, light="warm"):
        for row in range(rows):
            cy = height * (row + 1) / (rows + 1)
            for column in range(columns):
                u = (column - (columns - 1) / 2) * w * .6 / max(columns - 1, 1)
                v = (column - (columns - 1) / 2) * d * .6 / max(columns - 1, 1)
                if (row + column) % 7 == 0:
                    continue
                dx, dz, hy = w * .12, d * .12, min(.19, height / (rows + 1) * .23)
                self.quad((x+u-dx,cy-hy,z+d/2+.015),
                          (x+u+dx,cy-hy,z+d/2+.015),
                          (x+u+dx,cy+hy,z+d/2+.015),
                          (x+u-dx,cy+hy,z+d/2+.015),light)
                self.quad((x+u+dx,cy-hy,z-d/2-.015),
                          (x+u-dx,cy-hy,z-d/2-.015),
                          (x+u-dx,cy+hy,z-d/2-.015),
                          (x+u+dx,cy+hy,z-d/2-.015),light)
                self.quad((x+w/2+.015,cy-hy,z+v+dz),
                          (x+w/2+.015,cy-hy,z+v-dz),
                          (x+w/2+.015,cy+hy,z+v-dz),
                          (x+w/2+.015,cy+hy,z+v+dz),light)


def building_models(detail):
    models = {}

    m = Mesh()
    m.box(0,0,0,4.4,20,4.4,"glass_dark")
    m.window_rows(0,0,4.4,4.4,20,10 if detail else 6,3 if detail else 2,"cyan")
    m.box(0,20,0,3,3,3,"steel")
    m.box(0,23,0,1.6,3,1.6,"glass")
    m.cylinder(0,26,0,.24,4,"cyan",10 if detail else 7)
    m.cone(0,30,0,.52,1.4,"orange")
    models["central_spire"] = (m,"중앙 업무 지구 랜드마크")

    m = Mesh()
    m.box(0,0,0,4.5,16,3.5,"glass")
    m.window_rows(0,0,4.5,3.5,16,10 if detail else 6,3 if detail else 2)
    m.box(0,16,0,4.7,.25,3.7,"dark_steel")
    m.box(0,16.25,0,1,1.4,1,"steel")
    models["office_tower"] = (m,"유리 외벽 사무용 빌딩")

    m = Mesh()
    m.box(0,0,0,4.2,9.2,3.3,"cream")
    m.window_rows(0,0,4.2,3.3,9.2,7 if detail else 5,3 if detail else 2,"glass_dark")
    for level in (2.8,5.7,8.6):
        m.box(0,level,1.7,3.5,.1,.42,"concrete")
    m.box(0,9.2,0,4.5,.25,3.6,"dark_steel")
    models["apartment_block"] = (m,"주거 지구 아파트")

    m = Mesh()
    m.box(0,0,0,4.8,4.2,3.6,"steel")
    m.box(0,4.2,0,5,.24,3.8,"dark_steel")
    m.box(-1.1,4.44,.5,1.2,1.5,1.2,"concrete")
    for x in (.8,1.6):
        m.cylinder(x,4.44,-.8,.36,3.4,"rust",12 if detail else 8)
        m.cylinder(x,7.84,-.8,.44,.22,"dark_steel",12 if detail else 8)
    m.box(0,.05,1.82,2.7,2,.07,"black")
    models["factory"] = (m,"산업 구역 공장과 굴뚝")

    m = Mesh()
    m.box(0,0,0,5.2,3.2,4.2,"cream")
    m.box(0,3.2,0,5.5,.22,4.5,"dark_steel")
    for x in (-1.55,0,1.55):
        m.box(x,.05,2.12,1.2,2.45,.05,"glass_dark")
    models["warehouse"] = (m,"항만 창고")

    m = Mesh()
    for x in (-1.5,1.5):
        m.box(x,0,0,.42,1.2,.42,"steel")
        m.box(x,1.2,0,.38,1.5,.38,"dark_steel")
    m.cylinder(0,2.7,0,2.2,2.8,"concrete",14 if detail else 9)
    m.cylinder(0,5.5,0,2.25,.2,"steel",14 if detail else 9)
    m.box(0,5.7,0,.55,.8,.55,"cyan")
    models["industrial_tank"] = (m,"가스 저장 탱크")

    m = Mesh()
    m.box(-1.4,0,0,2.1,3.5,2.5,"steel")
    m.box(1.4,0,0,2.1,3.5,2.5,"steel")
    m.box(0,3.5,0,5.2,.35,3.1,"dark_steel")
    for x in (-2.35,2.35):
        m.box(x,3.85,0,.22,.85,.22,"orange")
    m.cylinder(0,3.85,0,.55,1.9,"concrete",8)
    models["power_station"] = (m,"전력 설비")

    return models


def environment_models(detail):
    models = {}
    m = Mesh()
    for x in (-2.1,2.1):
        m.box(x,0,0,.55,.35,1.2,"concrete")
        m.box(x,.35,0,.3,9.6,.3,"steel")
    m.beam((-2.1,9.5,0),(3.8,9.5,0),.31,"orange")
    m.beam((-2.1,8.8,0),(2.6,9.5,0),.17,"steel")
    m.box(2.3,8.8,0,.55,.55,.55,"dark_steel")
    m.beam((2.3,8.8,0),(2.3,3.2,0),.09,"black")
    m.box(2.3,2.8,0,.7,.4,.7,"orange")
    models["harbor_crane"] = (m,"항만 크레인")

    m = Mesh()
    for z, color in ((-.9,"blue"),(.9,"rust")):
        m.box(0,0,z,3.6,1.4,1.5,color)
        for x in (-1.45,-.7,.0,.7,1.45):
            m.box(x,.18,z+.76,.04,1.05,.03,"dark_steel")
    m.box(-.25,1.4,0,3.4,1.4,1.5,"cream")
    models["cargo_containers"] = (m,"쌓인 화물 컨테이너")

    m = Mesh()
    m.box(0,0,0,.38,1.65,.38,"trunk")
    m.cone(0,1.05,0,1.15,2.2,"green",10 if detail else 7)
    m.cone(0,2,0,.9,1.8,"green_light",10 if detail else 7)
    models["park_tree"] = (m,"공원 나무")

    m = Mesh()
    m.box(0,0,0,.14,2.5,.14,"steel")
    m.beam((0,2.45,0),(1.05,2.45,0),.13,"steel")
    m.box(1.05,2.29,0,.42,.16,.28,"warm")
    models["streetlamp"] = (m,"도로 가로등")

    m = Mesh()
    m.box(0,.27,0,1.05,.52,1.85,"blue")
    m.box(0,.79,-.15,.88,.42,1.05,"glass_dark")
    for x in (-.55,.55):
        for z in (-.54,.54):
            m.cylinder(x,.29,z,.3,.13,"black",10 if detail else 7,"x")
    m.box(0,.38,-.94,.75,.15,.05,"warm")
    m.box(0,.38,.94,.75,.15,.05,"red")
    models["compact_car"] = (m,"작은 도로 차량")

    m = Mesh()
    m.cylinder(0,0,0,1.65,.22,"concrete",16 if detail else 10)
    m.cylinder(0,.22,0,1.22,.22,"water",16 if detail else 10)
    m.cylinder(0,.44,0,.4,.75,"cream",10 if detail else 7)
    m.octahedron(0,1.23,0,.55,.52,.55,"cyan")
    models["park_fountain"] = (m,"공원 분수대")

    m = Mesh()
    m.box(0,0,0,7.2,.13,7.2,"asphalt")
    for z in (-2.4,-.8,.8,2.4):
        m.box(0,.135,z,.1,.015,.9,"cream")
    models["road_tile"] = (m,"도로 타일")
    return models


def effect_models(detail):
    models = {}
    m = Mesh()
    m.octahedron(0,0,0,1.3,1.5,1.2,"meteor")
    for x,y,z in ((.7,.7,.6),(-.8,-.1,.4),(.2,-.85,-.7)):
        m.octahedron(x,y,z,.4,.3,.4,"ember")
    models["meteor"] = (m,"운석 발사체 (중심 원점)")

    m = Mesh()
    m.octahedron(0,0,0,.7,.36,.55,"white")
    m.box(0,-.25,.05,.55,.13,.5,"cyan")
    for x in (-.83,.83):
        for z in (-.7,.7):
            m.beam((0,0,0),(x,0,z),.11,"steel")
            m.cylinder(x,.09,z,.38,.07,"dark_steel",10 if detail else 7)
    models["repair_drone"] = (m,"복구용 드론 (중심 원점)")
    return models


def write_glb(path: Path, name: str, mesh: Mesh):
    blob = bytearray()
    views, accessors, primitives, materials = [], [], [], []
    bounds = [float('inf')] * 3, [float('-inf')] * 3

    def add_array(values):
        while len(blob) % 4:
            blob.append(0)
        offset = len(blob)
        blob.extend(struct.pack('<' + 'f' * len(values), *values))
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(blob)-offset,
                      "target": 34962})
        return len(views) - 1

    for material, data in sorted(mesh.groups.items()):
        positions, normals = data['positions'], data['normals']
        if not positions:
            continue
        coords = [positions[i::3] for i in range(3)]
        low, high = [min(v) for v in coords], [max(v) for v in coords]
        for i in range(3):
            bounds[0][i] = min(bounds[0][i], low[i])
            bounds[1][i] = max(bounds[1][i], high[i])
        pv = add_array(positions)
        nv = add_array(normals)
        pi = len(accessors)
        accessors.append({"bufferView":pv,"componentType":5126,"count":len(positions)//3,
                          "type":"VEC3","min":low,"max":high})
        accessors.append({"bufferView":nv,"componentType":5126,"count":len(normals)//3,
                          "type":"VEC3"})
        hex_color, metal, rough = MATERIALS[material]
        rgb = [int(hex_color[i:i+2],16)/255 for i in (1,3,5)]
        materials.append({"name":material,"pbrMetallicRoughness":{
            "baseColorFactor":rgb+[1.0],"metallicFactor":metal,"roughnessFactor":rough},
            "doubleSided":True})
        primitives.append({"attributes":{"POSITION":pi,"NORMAL":pi+1},
                           "material":len(materials)-1,"mode":4})

    gltf = {"asset":{"version":"2.0","generator":"City Smash procedural asset builder"},
            "scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0,"name":name}],
            "meshes":[{"name":name,"primitives":primitives}],"materials":materials,
            "buffers":[{"byteLength":len(blob)}],"bufferViews":views,"accessors":accessors}
    json_chunk = json.dumps(gltf,separators=(',',':'),ensure_ascii=True).encode('utf-8')
    json_chunk += b' ' * (-len(json_chunk) % 4)
    blob.extend(b'\0' * (-len(blob) % 4))
    payload_length = 12 + 8 + len(json_chunk) + 8 + len(blob)
    path.write_bytes(struct.pack('<III',0x46546C67,2,payload_length) +
                     struct.pack('<I4s',len(json_chunk),b'JSON') + json_chunk +
                     struct.pack('<I4s',len(blob),b'BIN\0') + blob)
    return {"file":path.name,"triangles":sum(len(g['positions'])//9 for g in mesh.groups.values()),
            "bytes":path.stat().st_size,"bounds":{
                "min":[round(x,3) for x in bounds[0]],
                "max":[round(x,3) for x in bounds[1]]},
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


def generate(output: Path, quality: str):
    output.mkdir(parents=True,exist_ok=True)
    detail = quality == 'detailed'
    models = {}
    for family in (building_models(detail),environment_models(detail),effect_models(detail)):
        models.update(family)
    entries = []
    for name, (mesh, description) in sorted(models.items()):
        record = write_glb(output / (name + '.glb'),name,mesh)
        record['description'] = description
        entries.append(record)
    manifest = {"pack":"City Smash 3D Assets","format":"glTF 2.0 binary (GLB)",
                "quality":quality,"axes":"Y-up, metres","units":"metres",
                "coordinate_note":"Models rest on Y=0; meteor and drone are centred at the origin.",
                "assets":entries}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    verify(output)
    print(f"Generated {len(entries)} GLB models, {sum(e['bytes'] for e in entries)} bytes total in {output}")


def verify(output: Path):
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    entries = manifest['assets']
    actual = set(output.glob('*.glb'))
    expected = {output/item['file'] for item in entries}
    if len(entries) != 16 or actual != expected:
        raise ValueError('Expected exactly 16 GLB models and no stale files')
    for item in entries:
        data = (output/item['file']).read_bytes()
        magic, version, size = struct.unpack_from('<III',data)
        if (magic,version,size)!=(0x46546C67,2,len(data)):
            raise ValueError(f"Invalid GLB header: {item['file']}")
        json_size, chunk_type = struct.unpack_from('<I4s',data,12)
        if chunk_type != b'JSON':
            raise ValueError('Missing JSON chunk')
        gltf = json.loads(data[20:20+json_size])
        bin_size, bin_type = struct.unpack_from('<I4s',data,20+json_size)
        if bin_type != b'BIN\0' or 28+json_size+bin_size != len(data):
            raise ValueError('Missing or incomplete binary chunk')
        if gltf['asset']['version'] != '2.0' or not gltf['meshes'][0]['primitives']:
            raise ValueError('Missing glTF mesh')
        for view in gltf['bufferViews']:
            if view['byteOffset'] % 4 or view['byteOffset'] + view['byteLength'] > bin_size:
                raise ValueError('Invalid or unaligned buffer view')
        if item['triangles'] <= 0 or item['bytes'] != len(data):
            raise ValueError('Empty or incorrect asset record')
        if hashlib.sha256(data).hexdigest()!=item['sha256']:
            raise ValueError('Asset checksum mismatch')
        if manifest['quality']=='mobile' and len(data)>250_000:
            raise ValueError('Mobile GLB exceeded size budget')
    print(f"Verified {len(entries)} GLB models and checksums in {output}")


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output',type=Path,help='Folder to create the assets in')
    mode.add_argument('--verify',type=Path,help='Validate an existing output folder')
    parser.add_argument('--quality',choices=('mobile','detailed'),default='mobile')
    args=parser.parse_args()
    verify(args.verify) if args.verify else generate(args.output,args.quality)
