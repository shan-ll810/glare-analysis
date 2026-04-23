"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls, Edges } from "@react-three/drei";

type RoomPreviewProps = {
  length: number;
  width: number;
  height: number;
  windowHeight: number;
  hasShading: boolean;
  shadingDepth: number;
  shadingMode: string;
};

function RoomModel({
  length,
  width,
  height,
  windowHeight,
  hasShading,
  shadingDepth,
  shadingMode,
}: RoomPreviewProps) {
  const boxLength = Math.max(length / 10, 0.5);
  const boxWidth = Math.max(width / 10, 0.5);
  const boxHeight = Math.max(height / 10, 0.5);

  const winHeight = Math.min(Math.max(windowHeight / 10, 0.1), boxHeight * 0.8);
  const winWidth = boxLength * 0.45;

  // front wall
  const wallZ = boxWidth / 2 + 0.002;

  // window center
  const winCenterY = boxHeight * 0.55;

  // shading dimensions
  const shadeDepth = Math.max(shadingDepth / 10, 0.05);
  const shadeThickness = 0.03;

  // IMPORTANT:
  // since front wall is at +Z, outdoor shading should extend toward +Z
  const shadeCenterZ = wallZ + shadeDepth / 2;

  return (
    <group>
      {/* Room */}
      <mesh position={[0, boxHeight / 2, 0]}>
        <boxGeometry args={[boxLength, boxHeight, boxWidth]} />
        <meshStandardMaterial transparent opacity={0.15} />
        <Edges />
      </mesh>

      {/* Window */}
      <mesh position={[0, winCenterY, wallZ]}>
        <planeGeometry args={[winWidth, winHeight]} />
        <meshStandardMaterial transparent opacity={0.65} />
      </mesh>

      {/* Horizontal overhang */}
      {hasShading && shadingMode === "horizontal" && (
        <mesh
          position={[
            0,
            winCenterY + winHeight / 2 + shadeThickness / 2,
            shadeCenterZ,
          ]}
        >
          <boxGeometry args={[winWidth, shadeThickness, shadeDepth]} />
          <meshStandardMaterial />
        </mesh>
      )}

      {/* Left vertical fin */}
      {hasShading &&
        (shadingMode === "vertical-left" || shadingMode === "vertical-both") && (
          <mesh
            position={[
              -winWidth / 2 - shadeThickness / 2,
              winCenterY,
              shadeCenterZ,
            ]}
          >
            <boxGeometry args={[shadeThickness, winHeight, shadeDepth]} />
            <meshStandardMaterial />
          </mesh>
        )}

      {/* Right vertical fin */}
      {hasShading &&
        (shadingMode === "vertical-right" || shadingMode === "vertical-both") && (
          <mesh
            position={[
              winWidth / 2 + shadeThickness / 2,
              winCenterY,
              shadeCenterZ,
            ]}
          >
            <boxGeometry args={[shadeThickness, winHeight, shadeDepth]} />
            <meshStandardMaterial />
          </mesh>
        )}
    </group>
  );
}

export default function RoomPreview(props: RoomPreviewProps) {
  return (
    <div className="h-[320px] w-full overflow-hidden rounded-xl border bg-white">
      <Canvas camera={{ position: [2.5, 2, 2.5], fov: 50 }}>
        <ambientLight intensity={1.2} />
        <directionalLight position={[3, 4, 2]} intensity={1.2} />
        <RoomModel {...props} />
        <OrbitControls enablePan={false} />
      </Canvas>
    </div>
  );
}